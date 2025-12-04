import dataclasses
import math
from typing import List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

# Import the upstream PyTorch implementation
from timesfm import pytorch_patched_decoder as ppd


@dataclasses.dataclass
class TimesfmMRConfig(ppd.TimesFMConfig):
  """Config extension to toggle multi-resolution behaviors.

  - use_multi_resolution: add scale embeddings (low/high) to the token stream.
  - use_special_token_s: insert a learned special token between streams.
  - use_multi_task / horizon_len_hr: optional extra head for a second horizon.
  """

  use_multi_resolution: bool = False
  use_special_token_s: bool = False
  use_multi_task: bool = False
  horizon_len_hr: int = 128


class PatchedTSMultiResolutionDecoder(ppd.PatchedTimeSeriesDecoder):
  """Extension of upstream decoder with multi-resolution support.

  This class keeps the upstream API intact, while enabling two optional
  behaviors:
    - scale embedding per token for low/high streams,
    - an optional learned special token between streams.
  """

  def __init__(self, config: TimesfmMRConfig):
    super().__init__(config)
    self.config: TimesfmMRConfig

    # Multi-resolution Embedding Layer
    if self.config.use_multi_resolution:
      self.multi_resolution = nn.Embedding(num_embeddings=2,
                                           embedding_dim=self.config.hidden_size)

    # Special Token between streams
    if self.config.use_special_token_s:
      self.special_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
      nn.init.normal_(self.special_token, mean=0.0, std=0.02)

    # Optional extra head for multi-task horizon
    if self.config.use_multi_task:
      self.horizon_ff_layer_hr = ppd.ResidualBlock(
          input_dims=self.config.hidden_size,
          output_dims=self.config.horizon_len_hr * (1 + len(self.config.quantiles)),
          hidden_dims=self.config.intermediate_size,
      )

  
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
    num_outputs = len(self.config.quantiles) + 1

    if isinstance(input_ts, torch.Tensor):
      # PATH 1: Single resolution (when we're not using multi-resolution approaches | also upstream compatible with decode/forecast)
      model_input, patched_padding, stats, _ = super()._preprocess_input(
          input_ts=input_ts,
          input_padding=input_padding,
      )

      # Multi-resolution (single stream -> index 0)
      if self.config.use_multi_resolution:
        mr_idx = torch.zeros((model_input.shape[0], model_input.shape[1]),
                             dtype=torch.long,
                             device=model_input.device)
        mr_emb = self.multi_resolution(mr_idx)
        model_input = model_input + mr_emb

      # Frequency embedding and transformer
      f_emb = self.freq_emb(freq)
      model_input = model_input + f_emb
      model_output = self.stacked_transformer(model_input, patched_padding)

      # Post-processing
      output_ts_min = self._postprocess_output(
          model_output=model_output,
          horizon_len=self.config.horizon_len,
          head=self.horizon_ff_layer,
          num_outputs=num_outputs,
          stats_list=[stats],
          indices_list=[(0, model_input.shape[1])],
      )

      # Additional head for multi-task (if enabled)
      if self.config.use_multi_task:
        output_ts_hr = self._postprocess_output(
            model_output=model_output,
            horizon_len=self.config.horizon_len_hr,
            head=self.horizon_ff_layer_hr,
            num_outputs=num_outputs,
            stats_list=[stats],
            indices_list=[(0, model_output.shape[1])],
        )

        return {"min": output_ts_min, "hr": output_ts_hr}

      return output_ts_min

    # PATH 2: Multi-resolution (when we're using multi-resolution approaches)
    ts_low, ts_high = input_ts
    pad_low, pad_high = input_padding

    model_input_1, pad_1, stats_1, _ = super()._preprocess_input(
        input_ts=ts_low,
        input_padding=pad_low,
    )
    model_input_2, pad_2, stats_2, _ = super()._preprocess_input(
        input_ts=ts_high,
        input_padding=pad_high,
    )

    B = model_input_1.shape[0]
    Nlow = model_input_1.shape[1]
    Nhigh = model_input_2.shape[1]
    D = model_input_1.shape[2]
    device = model_input_1.device

    # Special Token between streams
    if self.config.use_special_token_s:
      spec_tok = self.special_token.to(device).expand(B, 1, D)
      spec_pad = torch.zeros(B, 1, device=device, dtype=pad_1.dtype)  # not padded

      model_input = torch.cat([model_input_1, spec_tok, model_input_2], dim=1)     # [B, N1+1+N2, D]
      patched_padding = torch.cat([pad_1, spec_pad, pad_2], dim=1)
      
      # Keep mask to drop the special token position after decoding
      keep_mask = torch.ones(Nlow + 1 + Nhigh, device=device, dtype=torch.bool)
      keep_mask[Nlow] = False  # special token index
      spec_len = 1
    else:
      model_input = torch.cat([model_input_1, model_input_2], dim=1)               # [B, N1+N2, D]
      patched_padding = torch.cat([pad_1, pad_2], dim=1)
      keep_mask = None
      spec_len = 0

    # Multi-resolution Embedding
    if self.config.use_multi_resolution:
      mr_first = torch.zeros(Nlow, dtype=torch.long, device=device)
      mr_spec = torch.zeros(spec_len, dtype=torch.long, device=device)              # use 0 for special token
      mr_second = torch.ones(Nhigh, dtype=torch.long, device=device)

      mr_idx = torch.cat([mr_first, mr_spec, mr_second], dim=0)                     # [N_total]
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
        (0, Nlow),
        (Nlow + spec_len, Nlow + spec_len + Nhigh),
    ]
    stats_list = [stats_1, stats_2]

    output_min_all = self._postprocess_output(
        model_output=model_output,
        horizon_len=self.config.horizon_len,
        head=self.horizon_ff_layer,
        num_outputs=num_outputs,
        stats_list=stats_list,
        indices_list=indices_list,
    )

    if self.config.use_multi_task:
      output_hr_all = self._postprocess_output(
          model_output=model_output,
          horizon_len=self.config.horizon_len_hr,
          head=self.horizon_ff_layer_hr,
          num_outputs=num_outputs,
          stats_list=[stats_1],
          indices_list=[(0, Nlow + spec_len + Nhigh)],
      )

    if keep_mask is not None:
      output_min_all = output_min_all[:, keep_mask, :, :]

      if self.config.use_multi_task:
        output_hr_all = output_hr_all[:, keep_mask, :, :]

    if self.config.use_multi_task:
      output_hr_low = output_hr_all[:, :Nhigh, :, :]     # hour forecasts for low-res patches
      output_min_high = output_min_all[:, Nlow:, :, :]    # minute forecasts for high-res patches
      return {"min": output_min_high, "hr": output_hr_low}
    else:
      return output_min_all


  def decode(
      self,
      input_ts: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
      paddings: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
      freq: torch.LongTensor,
      horizon_len: int,
      output_patch_len: int | None = None,
      agg_factor: int = 60
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-regressive decoding for single or multi-resolution inputs.

    Mirrors upstream behavior while supporting two-stream inputs where the
    low-resolution stream is kept fixed and the high-resolution stream is
    autoregressively generated.

    Args:
      input_ts: BxC tensor for single-res, or [ts_low, ts_high] tensors.
      paddings: Bx(C+H) tensor, or [pad_low, pad_high] tensors.
      freq: Bx1 frequency tensor.
      horizon_len: prediction horizon.
      output_patch_len: step size per decode iteration.
      agg_factor: aggregation factor from high-res to low-res.

    Returns:
      Tuple(mean, full):
        mean: B x H' tensor of mean forecasts.
        full: B x H' x (1 + num_quantiles) full forecasts.
    """
    t_low, t_high = input_ts
    p_low, p_high = paddings

    final_out_high = t_high
    full_outputs: list[torch.Tensor] = []
    context_len_high = int(t_high.shape[1])
    max_len = context_len_high

    output_patch_len = output_patch_len or horizon_len
    num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len

    for step_index in range(num_decode_patches):
      current_pad_high = p_high[:, 0:final_out_high.shape[1]]
      # Trim to window
      in_low = t_low[:, -max_len:]
      in_pad_low = p_low[:, -max_len:]
      in_high = final_out_high[:, -max_len:]
      in_pad_high = current_pad_high[:, -max_len:]

      fprop_outputs = self([in_low, in_high], [in_pad_low, in_pad_high], freq)
      if isinstance(fprop_outputs, dict):
        fprop_outputs = fprop_outputs.get("min")

      # Keep full last-patch for return aggregation
      new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
      full_outputs.append(new_full_ts)

      # High-res: append ALL predictions from this patch (no overlap)
      new_high_all = new_full_ts[:, :, 0]  # [B, output_patch_len]
      final_out_high = torch.concatenate([final_out_high, new_high_all], dim=-1)
      p_high = torch.concatenate(
          [p_high,
           torch.zeros((p_high.shape[0], new_high_all.shape[1]), device=p_high.device, dtype=p_high.dtype)],
          dim=-1)

      # Low-res: aggregate only the first `picked` minutes, where
      # `picked` is the largest multiple of agg_factor <= output_patch_len. If none, skip.
      picked = int((int(output_patch_len) // int(agg_factor)) * int(agg_factor))
      picked = min(picked, new_full_ts.shape[1]) if picked > 0 else 0
      if picked > 0:
        ts_high_for_agg = new_full_ts[:, :picked, 0]
        num_low = picked // agg_factor
        if num_low > 0:
          used = num_low * agg_factor
          low_blocks = ts_high_for_agg[:, :used].reshape(ts_high_for_agg.shape[0], num_low, agg_factor)
          ts_low_chunk = low_blocks.mean(dim=-1)
          t_low = torch.concatenate([t_low, ts_low_chunk], dim=-1)
          p_low = torch.concatenate(
              [p_low,
               torch.zeros((p_low.shape[0], num_low), device=p_low.device, dtype=p_low.dtype)],
              dim=-1)

    full_cat = torch.concatenate(full_outputs, dim=1)[:, 0:horizon_len, :]

    return (full_cat[:, :, 0], full_cat)
