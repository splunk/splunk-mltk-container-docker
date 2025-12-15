#
# Copyright 2025 Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import dataclasses
from typing import List, Tuple, Union

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from timesfm import pytorch_patched_decoder as ppd


@dataclasses.dataclass
class CiscoTsmMRConfig(ppd.TimesFMConfig):
  """Config extension to toggle multi-resolution behaviors.

  - use_resolution_embeddings: add scale embeddings (low/high) to the token stream.
  - use_special_token: insert a learned special token between streams.
  """

  use_resolution_embeddings: bool = False
  use_special_token: bool = False


class PatchedTSMultiResolutionDecoder(ppd.PatchedTimeSeriesDecoder):
  """Extension of upstream decoder with multi-resolution support.

  This class keeps the upstream API intact, while enabling two optional
  behaviors:
    - scale embedding per token for low/high streams,
    - an optional learned special token between streams.
  """

  def __init__(self, config: CiscoTsmMRConfig):
    super().__init__(config)
    self.config: CiscoTsmMRConfig

    # Multi-resolution Embedding Layer
    if self.config.use_resolution_embeddings:
      self.multi_resolution = nn.Embedding(num_embeddings=2,
                                           embedding_dim=self.config.hidden_size)

    # Special Token between streams
    if self.config.use_special_token:
      self.special_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
      nn.init.normal_(self.special_token, mean=0.0, std=0.02)

  
  def _reverse_transform_segments(
      self,
      outputs: torch.Tensor,
      stats_list: List[Tuple[torch.Tensor, torch.Tensor]],
      indices_list: List[Tuple[int, int]],
  ) -> torch.Tensor:
    """Reverse-transform with per-timeseries stats.

    Args:
      outputs: [B, N, P, Q]
      stats_list: list of (mu, sigma) each shaped [B]
      indices_list: matching list of (start_N, end_N) segment ranges over N
    """
    B, N, _, _ = outputs.shape
    device = outputs.device
    dtype = outputs.dtype

    if len(indices_list) == 0:
      return outputs

    # Build [S] tensors of segment starts/ends (S = number of streams)
    starts = torch.tensor([s for (s, _) in indices_list], device=device)
    ends = torch.tensor([e for (_, e) in indices_list], device=device)
    S = starts.shape[0]

    # Per-batch stats stacked as [B, S]
    mus = torch.stack([mu.to(dtype) for (mu, _) in stats_list], dim=1)       # [B, S]
    sigmas = torch.stack([sigma.to(dtype) for (_, sigma) in stats_list], dim=1)  # [B, S]

    # Build boolean mask per segment over N: [S, N]
    posN = torch.arange(N, device=device)
    seg_mask_SN = ((posN.unsqueeze(0) >= starts.unsqueeze(1)) &
                   (posN.unsqueeze(0) < ends.unsqueeze(1)))  # [S, N]

    # Expand to broadcast shapes:
    # seg_mask: [1, S, N, 1, 1], mus/sigmas: [B, S, 1, 1, 1]
    seg_mask = seg_mask_SN.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(dtype)  # [1, S, N, 1, 1]
    mus_b = mus.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                      # [B, S, 1, 1, 1]
    sigmas_b = sigmas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                # [B, S, 1, 1, 1]

    # Aggregate per-position parameters
    mu_map = (mus_b * seg_mask).sum(dim=1)                                     # [B, N, 1, 1]
    sigma_map = (sigmas_b * seg_mask).sum(dim=1)                                # [B, N, 1, 1]

    # For positions not covered by any segment, keep outputs unchanged: sigma=1, mu=0
    covered = (seg_mask.sum(dim=1) > 0).to(dtype)                               # [1, N, 1, 1]
    sigma_map = sigma_map + (1.0 - covered).expand(B, -1, -1, -1)                # add 1 where uncovered

    return outputs * sigma_map + mu_map


  def _postprocess_output(
      self,
      model_output: torch.Tensor,
      horizon_len: int,
      head: nn.Module,
      num_outputs: int,
      stats_list: list[tuple[torch.Tensor, torch.Tensor]],
      indices_list: list[tuple[int, int]],
  ) -> torch.Tensor:
    """Postprocess output of stacked transformer."""

    # B x N x (H.Q)
    output_ts = head(model_output)

    # Reshape using view
    b, n, _ = output_ts.shape
    output_ts = output_ts.view(b, n, horizon_len, num_outputs)

    return self._reverse_transform_segments(output_ts, stats_list, indices_list)

  
  def forward(
      self,
      input_ts: Union[List[torch.Tensor], torch.Tensor],
      input_padding: Union[List[torch.LongTensor], torch.LongTensor],
      freq: torch.Tensor,
  ) -> torch.Tensor:
    """Multi-resolution forward pass.
    Args:
      input_ts: list of batched tensors for coarse/fine resolution streams.
      input_padding: list of batched paddings for coarse/fine resolution streams.
      freq: batched tensor of frequency indices.
    """
    num_outputs = len(self.config.quantiles) + 1

    if isinstance(input_ts, torch.Tensor):
      raise ValueError("PatchedTSMultiResolutionDecoder expects multi-resolution inputs as a list of tensors.")

    # Multi-resolution processing
    ts_coarse, ts_fine = input_ts
    pad_coarse, pad_fine = input_padding

    model_input_coarse, pad_coarse, stats_coarse, _ = super()._preprocess_input(
        input_ts=ts_coarse,
        input_padding=pad_coarse,
    )
    model_input_fine, pad_fine, stats_fine, _ = super()._preprocess_input(
        input_ts=ts_fine,
        input_padding=pad_fine,
    )

    B = model_input_coarse.shape[0]
    Ncoarse = model_input_coarse.shape[1]
    Nfine = model_input_fine.shape[1]
    D = model_input_coarse.shape[2]
    device = model_input_coarse.device

    # Special Token between streams
    if self.config.use_special_token:
      spec_tok = self.special_token.to(device).expand(B, 1, D)
      spec_pad = torch.zeros(B, 1, device=device, dtype=pad_coarse.dtype)

      model_input = torch.cat([model_input_coarse, spec_tok, model_input_fine], dim=1)     # [B, N1+1+N2, D]
      patched_padding = torch.cat([pad_coarse, spec_pad, pad_fine], dim=1)
      
      # Keep mask to drop the special token position after decoding
      keep_mask = torch.ones(Ncoarse + 1 + Nfine, device=device, dtype=torch.bool)
      keep_mask[Ncoarse] = False  # special token index
      spec_len = 1
    else:
      model_input = torch.cat([model_input_coarse, model_input_fine], dim=1)               # [B, N1+N2, D]
      patched_padding = torch.cat([pad_coarse, pad_fine], dim=1)
      keep_mask = None
      spec_len = 0

    # Multi-resolution Embedding
    if self.config.use_resolution_embeddings:
      mr_coarse = torch.zeros(Ncoarse, dtype=torch.long, device=device)
      mr_spec = torch.zeros(spec_len, dtype=torch.long, device=device)              # we use 0 for special token
      mr_fine = torch.ones(Nfine, dtype=torch.long, device=device)

      mr_idx = torch.cat([mr_coarse, mr_spec, mr_fine], dim=0)                     # [N_total]
      mr_idx = mr_idx.unsqueeze(0).expand(B, -1)                                    # [B, N_total]
      
      mr_emb = self.multi_resolution(mr_idx)                                        # [B, N_total, D]
      model_input += mr_emb

    if freq.device != device:
      freq = freq.to(device)
    
    f_emb = self.freq_emb(freq)  # [B, 1, D]
    model_input += f_emb

    model_output = self.stacked_transformer(model_input, patched_padding)

    # Project and apply per-segment reverse-transform

    indices_list = [
        (0, Ncoarse),
        (Ncoarse + spec_len, Ncoarse + spec_len + Nfine),
    ]
    stats_list = [stats_coarse, stats_fine]

    output_min_all = self._postprocess_output(
        model_output=model_output,
        horizon_len=self.config.horizon_len,
        head=self.horizon_ff_layer,
        num_outputs=num_outputs,
        stats_list=stats_list,
        indices_list=indices_list,
    )

    if keep_mask is not None:
      output_min_all = output_min_all[:, keep_mask, :, :]

    return output_min_all


  def _trim_context(self, ts: torch.Tensor, pad: torch.Tensor, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim context to be aligned to patch boundaries and within max length.
    Args:
      ts: [B, T, C] input time series tensor.
      pad: [B, T, 1] input padding tensor.
      max_len: maximum allowed length.
    Returns:
      Trimmed (ts, pad) tensors.
    """
    target_len = max(self.config.patch_len, (max_len // self.config.patch_len) * self.config.patch_len)
    
    if ts.shape[1] > target_len:
      ts = ts[:, -target_len:, :]
      pad = pad[:, -target_len:, :]
    
    rem = ts.shape[1] % self.config.patch_len
    
    if rem:
      ts = ts[:, rem:, :]
      pad = pad[:, rem:, :]
    
    return ts, pad
  

  def decode(
      self,
      input_ts: Union[list[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
      paddings: Union[list[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
      freq: torch.LongTensor,
      horizon_len: int,
      agg_factor: int = 60,
      offsets: List[float] = None,
      scales: List[float] = None,
      global_eps: float = 1e-8,
      output_patch_len: int = 128,
  ) -> List[Tuple[List[float], dict[str, List[float]]]]:
    """Autoregressive Multiresolution Decoding.

    Args:
      input_ts: list of [B, T1, C], [B, T2, C] tensors for low/high resolution streams.
      paddings: list of [B, T1, 1], [B, T2, 1] paddings for low/high resolution streams.
      freq: [B] tensor of frequency indices.
      horizon_len: total forecast horizon length (in high-res steps).
      agg_factor: aggregation factor from high-res to low-res (e.g., 60 for min->hr).
      offsets: list of length B of denormalization offsets for high-res stream.
      scales: list of length B of denormalization scales for high-res stream.
      global_eps: small value to avoid division by zero during denormalization.
      output_patch_len: number of high-res steps to decode per iteration.

    Returns:
      A list of length B of tuples:
        - mean forecast list of length `horizon_len`,
        - dict of quantile forecasts, each a list of length `horizon_len`.
    """
    if agg_factor <= 0:
      raise ValueError("agg_factor must be positive for autoregressive decoding.")

    q_levels = self.config.quantiles
    expected_q_plus_mean = len(q_levels) + 1
    expected_q_only = len(q_levels)

    if not isinstance(input_ts, (list, tuple)) or len(input_ts) != 2:
      raise ValueError("Multi-resolution autoregressive decoding expects [low_res, high_res] inputs.")

    coarse_ts, fine_ts = input_ts
    coarse_pad, fine_pad = paddings

    # Ensure 3D shapes
    if coarse_ts.ndim == 2:
      coarse_ts = coarse_ts.unsqueeze(-1)
    if fine_ts.ndim == 2:
      fine_ts = fine_ts.unsqueeze(-1)
    if coarse_pad.ndim == 2:
      coarse_pad = coarse_pad.unsqueeze(-1)
    if fine_pad.ndim == 2:
      fine_pad = fine_pad.unsqueeze(-1)

    device = fine_ts.device
    batch_size = fine_ts.shape[0]
    patch_len = self.config.patch_len
    output_patch_len = output_patch_len or self.config.horizon_len

    # Offsets/scales for denormalization
    offsets = [0.0] * batch_size if offsets is None else offsets
    scales = [1.0] * batch_size if scales is None else scales
    batch_offsets = torch.as_tensor(offsets, dtype=torch.float32, device=device).view(batch_size, 1)
    batch_scales = torch.as_tensor(scales, dtype=torch.float32, device=device).view(batch_size, 1)

    # Keep working windows aligned and trimming to patch boundaries for performing decoding step.
    max_ctx_len_coarse = max(patch_len, (coarse_ts.shape[1] // patch_len) * patch_len)
    coarse_ts, coarse_pad = self._trim_context(coarse_ts, coarse_pad, max_ctx_len_coarse)
    max_ctx_len_fine = max(patch_len, (fine_ts.shape[1] // patch_len) * patch_len)
    fine_ts, fine_pad = self._trim_context(fine_ts, fine_pad, max_ctx_len_fine)

    remaining = horizon_len
    mean_chunks = []
    quant_chunks = []
    
    # Number of decode steps to perform
    num_decode_patches = math.ceil(horizon_len / output_patch_len)

    for _ in range(num_decode_patches):
      preds = self([coarse_ts, fine_ts], [coarse_pad.float(), fine_pad.float()], freq)
      if preds.ndim != 4:
        raise ValueError(f"Unexpected prediction rank: {preds.shape}")

      num_coarse_patches = coarse_ts.shape[1] // patch_len
      num_fine_patches = fine_ts.shape[1] // patch_len
      fine_patch_idx = num_coarse_patches + num_fine_patches - 1
      
      if fine_patch_idx >= preds.shape[1]:
        raise ValueError(f"Fine patch index {fine_patch_idx} out of range for preds shape {preds.shape}")

      fine_patch = preds[:, fine_patch_idx, :, :]
      
      if fine_patch.shape[1] < output_patch_len:
        raise ValueError(f"Model horizon - {fine_patch.shape[1]} < requested output_patch_len {output_patch_len}")
      
      fine_patch = fine_patch[:, :output_patch_len, :]

      C = fine_patch.shape[2]
      if C == expected_q_plus_mean:
        mean_channel = fine_patch[..., 0]      # [B, L]
        quant_block = fine_patch[..., 1:]      # [B, L, Q]
      elif C == expected_q_only:
        mean_channel = None
        quant_block = fine_patch
      else:
        raise ValueError(f"Channel count {C} != {expected_q_plus_mean} or {expected_q_only}")

      if mean_channel is None:
        mean_channel = quant_block.median(dim=-1).values  # [B, L]

      step_taken = min(remaining, output_patch_len)
      mean_denorm = mean_channel * (batch_scales + global_eps) + batch_offsets
      quant_denorm = quant_block * (batch_scales.unsqueeze(-1) + global_eps) + batch_offsets.unsqueeze(-1)
      mean_denorm = torch.nan_to_num(mean_denorm, nan=0.0, posinf=0.0, neginf=-0.0)
      quant_denorm = torch.nan_to_num(quant_denorm, nan=0.0, posinf=0.0, neginf=-0.0)

      mean_chunks.append(mean_denorm[:, :step_taken])
      quant_chunks.append(quant_denorm[:, :step_taken, :])
      remaining -= step_taken

      # Append normalized minute predictions for the next step.
      fine_append = mean_channel[:, :output_patch_len].unsqueeze(-1)
      fine_ts = torch.cat([fine_ts, fine_append], dim=1)
      fine_pad = torch.cat(
        [fine_pad, torch.zeros((batch_size, output_patch_len, 1), device=device, dtype=fine_pad.dtype)],
        dim=1)

      # Aggregate minute predictions into coarse stream (drop remainder < agg_factor).
      agg_block_len = (output_patch_len // agg_factor) * agg_factor
      if agg_block_len > 0:
        agg_source = mean_channel[:, :agg_block_len]
        agg_vals = agg_source.view(batch_size, -1, agg_factor).mean(dim=2).unsqueeze(-1)
        coarse_ts = torch.cat([coarse_ts, agg_vals.to(coarse_ts.dtype)], dim=1)
        coarse_pad = torch.cat(
            [coarse_pad, torch.zeros((batch_size, agg_vals.shape[1], 1), device=device, dtype=coarse_pad.dtype)],
            dim=1)

      # Keep contexts aligned and bounded.
      fine_ts, fine_pad = self._trim_context(fine_ts, fine_pad, max_ctx_len_fine)
      coarse_ts, coarse_pad = self._trim_context(coarse_ts, coarse_pad, max_ctx_len_coarse)

      if remaining <= 0:
        break

    mean_full = torch.cat(mean_chunks, dim=1)[:, :horizon_len]
    quant_full = torch.cat(quant_chunks, dim=1)[:, :horizon_len, :]

    mean_np = mean_full.cpu().numpy()
    quant_np = quant_full.cpu().numpy()

    final_predictions = []
    for i in range(batch_size):
      q_arr = np.transpose(quant_np[i], (1, 0))  # [Q, H]
      final_predictions.append(
          {
            "mean": mean_np[i],
            "quantiles": {str(q_levels[q_i]): q_arr[q_i] for q_i in range(q_arr.shape[0])}
          }
      )
    
    return final_predictions
