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
import logging
from os import path
from typing import Any, List, Sequence, Union, Tuple

import numpy as np

import torch

from huggingface_hub import snapshot_download

from timesfm import TimesFmHparams, TimesFmCheckpoint
from timesfm.timesfm_torch import TimesFmTorch
from timesfm.timesfm_base import strip_leading_nans, linear_interpolation

from .patched_decoder_multi_resolution import CiscoTsmMRConfig, PatchedTSMultiResolutionDecoder


class CiscoTsmMR(TimesFmTorch):
  """Cisco Time Series Model Multi-resolution Forecast API."""

  def __init__(
      self,
      hparams: TimesFmHparams,
      checkpoint: TimesFmCheckpoint,
      *,
      use_resolution_embeddings: bool = True,
      use_special_token: bool = True,
  ) -> None:
    self.use_resolution_embeddings = use_resolution_embeddings
    self.use_special_token = use_special_token
    super().__init__(hparams, checkpoint)

  def __post_init__(self):
    # Building MR config
    self._model_config = CiscoTsmMRConfig(
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        hidden_size=self.model_dims,
        intermediate_size=self.model_dims,
        patch_len=self.input_patch_len,
        horizon_len=self.output_patch_len,
        head_dim=self.model_dims // self.num_heads,
        quantiles=self.quantiles,
        use_positional_embedding=self.use_pos_emb,
        use_resolution_embeddings=self.use_resolution_embeddings,
        use_special_token=self.use_special_token,
    )
    self._model = None
    self.num_cores = 1
    self.global_batch_size = self.per_core_batch_size
    self._device = torch.device("cuda:0" if (
        torch.cuda.is_available() and self.backend == "gpu") else "cpu")
    self._median_index = -1

  
  def load_from_checkpoint(
      self,
      checkpoint: TimesFmCheckpoint,
  ) -> None:
    """Loads a Multiresolution Model checkpoint from path and prepares the MR decoder for inference.
    Args:
      checkpoint: TimesFmCheckpoint object containing checkpoint info (local or HF repo).
    """

    checkpoint_path = checkpoint.path
    repo_id = checkpoint.huggingface_repo_id
    if checkpoint_path is None:
      checkpoint_path = path.join(
          snapshot_download(repo_id, local_dir=checkpoint.local_dir),
          "torch_model.pt")
    self._model = PatchedTSMultiResolutionDecoder(self._model_config)
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=self._device)
    logging.info("Loading checkpoint from %s", checkpoint_path)
    incompatible = self._model.load_state_dict(loaded_checkpoint, strict=True)

    if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
      logging.info(
          "MR decoder state load differences. missing=%s unexpected=%s",
          getattr(incompatible, "missing_keys", []),
          getattr(incompatible, "unexpected_keys", []),
      )

    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info("Sending checkpoint to device %s", f"{self._device}")

    self._model.to(self._device)
    self._model.eval()


  def _pad_or_truncate(self, 
                       ts: torch.Tensor, 
                       target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate a time series to a target length, especially [LEFT-PADDING].
    Args:
      ts: 1D or 2D tensor of shape [L] or [L, 1].
      target_len: desired target length after padding/truncation.
    Returns:
      padded_ts: tensor of shape [target_len].
      pad_mask: tensor of shape [target_len], with 1.0 for padded positions and 0.0 for actual data.
    
    """

    if ts.ndim == 2 and ts.shape[-1] == 1:
        ts = ts.squeeze(-1)
    
    L = ts.shape[0]
    
    if L == target_len:
        return ts, torch.zeros_like(ts, dtype=torch.float32)

    if L > target_len:
        return ts[-target_len:], torch.zeros(target_len, dtype=torch.float32)
    
    pad_len = target_len - L
    padded = torch.cat([torch.zeros(pad_len, dtype=ts.dtype), ts], dim=0)
    
    pad_mask = torch.cat([
        torch.ones(pad_len, dtype=torch.float32),
        torch.zeros(L, dtype=torch.float32)
    ], dim=0)
    
    return padded, pad_mask

  
  def normalize_with_pad(self, 
                         context, 
                         pad_mask: torch.Tensor | None = None, 
                         clamp_range=(-1000, 1000)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Normalize context with padding mask.
    Args:
      context: tensor of shape [B, T] or [B, T, 1].
      pad_mask: tensor of shape [B, T], with 1.0 for padded positions and 0.0 for actual data.
      clamp_range: tuple of (min, max) to clamp normalized values. Default (-1000, 1000).
    Returns:
      ctx_normalized: normalized context tensor with same shape as input.
      offset: mean used for normalization, shape [B, 1].
      scale: stddev used for normalization, shape [B, 1].
      eps: small epsilon value used for numerical stability.
    """

    eps = 1e-8

    if context.ndim == 3:
      context = context.squeeze(-1)

    if pad_mask is None:
      pad_mask = torch.zeros_like(context)

    valid = (1.0 - pad_mask)  # 1 for real, 0 for pad
    # Prevent divide-by-zero
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    # Masked mean, variance and std
    context_mean = (context * valid).sum(dim=1, keepdim=True) / count

    # Center for variance
    context_var = (((context - context_mean) * valid)**2).sum(dim=1, keepdim=True) / count
    context_std = context_var.sqrt()

    ctx_normalized = (context - context_mean) / (context_std + eps)
    stats = (context_mean, context_std)

    ctx_normalized = ctx_normalized * valid

    ctx_normalized = torch.clamp(ctx_normalized, *clamp_range)

    offset, scale = stats

    return ctx_normalized, offset, scale, eps
  

  def slice_fine_context(self, series: List[float], fine_len: int = 512) -> List[float]:
    """Return the rightmost fine_len points (or entire series if shorter).
    Args:
      series: list or array of fine-resolution (fine-level) time series data.
      fine_len: desired length of fine-level context to extract.
    Returns:
      List of floats representing the fine-level context of length <= fine_len.
    """
    return series[-fine_len:]
  
  
  def build_coarse_context(self, series: np.ndarray, max_coarse_ctx: int = 512, block: int = 60) -> List[float]:
    """Construct coarse context by:
       1. Taking up to rightmost (max_coarse_ctx * block) raw fine samples.
       2. Partitioning into consecutive non-overlapping blocks of 'block' size from left to right (chronological order preserved).
       3. Computing the mean of each block.
    
    Args:
      series: array of fine-resolution (fine-level) time series data.
      max_coarse_ctx: maximum number of coarse points to return.
      block: number of fine samples to aggregate into one coarse sample.
    Returns:
      List of floats representing coarse means with length <= max_coarse_ctx.
    """
    needed_raw = max_coarse_ctx * block
    raw_slice = series[-needed_raw:]
    # Ensure we only form full blocks; drop partial leading block if length not multiple
    remainder = len(raw_slice) % block
    if remainder != 0:
      raw_slice = raw_slice[remainder:]  # align to block boundary at the right edge
    coarse = []
    for i in range(0, len(raw_slice), block):
      block_vals = raw_slice[i:i+block]
      if len(block_vals) < block:
          break
      coarse.append(float(sum(block_vals) / block))
    return coarse[-max_coarse_ctx:]

  
  def build_multi_resolution(self, series: np.ndarray, agg_factor: int = 60) -> Tuple[List[float], List[float]]:
    """Builds multi-resolution contexts from a fine-resolution time series.
    Args:
      series: array of fine-resolution (fine-level) time series data.
      agg_factor: aggregation factor to form coarse context from fine context.
    Returns:
      Tuple of:
      - coarse_ctx: list of floats representing the coarse context.
      - fine_ctx: list of floats representing the fine context.
    """

    coarse_ctx = self.build_coarse_context(series, max_coarse_ctx=512, block=agg_factor)
    fine_ctx = self.slice_fine_context(series)
    return coarse_ctx, fine_ctx
  

  def _normalize_inputs(self, inputs):
    """Normalizes input series into a batched series.
    Args:
      inputs: single series or list of series.
    Returns:
      List of series, each as a list of floats.
    """

    if isinstance(inputs, (np.ndarray, torch.Tensor)):
      inputs = inputs.tolist()

    if not isinstance(inputs, (list, tuple)):
      return [[inputs]]

    # Compute series depth.
    def _depth(x):
      if isinstance(x, (list, tuple, np.ndarray, torch.Tensor)):
        return 1 + (max((_depth(y) for y in x), default=0))
      return 0

    d = _depth(inputs)
    if d > 2:
      raise ValueError("Input series must be strictly list-of-lists or a list.")

    if d == 1:
      return [list(inputs)]

    final_inps = []
    for s in inputs:
      if isinstance(s, (np.ndarray, torch.Tensor)):
        s = s.tolist()
      elif not isinstance(s, (list, tuple, np.ndarray, torch.Tensor)):
        raise ValueError("Each series must be list-like when providing a list of series.")
      
      final_inps.append(list(s))
    return final_inps

  
  def forecast(self, 
               inputs: Sequence[Any],
               horizon_len: Union[int, None] = None, 
               agg_factor: int = 60, 
               batch_size: int = 8) -> List[dict[str, Any]]:
    """Forecasts from a single fine-resolution stream.

    Derives the coarse-resolution stream by aggregating the fine-resolution
    context in blocks of `agg_factor` (e.g., 60 minutes -> hourly) and then
    runs multi-resolution decoding.

    Args:
      inputs: list-like of fine-resolution context series.
      horizon_len: forecast horizon length; if None, uses the model's configured horizon length.
      agg_factor: size of aggregation window to form the coarse context from the fine context.
      batch_size: batch size for forecasting.

    Returns:
      List of dictionaries containing mean and quantile forecasts for each series input.
    """
    if self._model is None:
      raise ValueError("Checkpoint is not properly loaded.")
    
    if horizon_len is None:
      horizon_len = self.output_patch_len

    if horizon_len <= 0:
      raise ValueError("horizon_len must be positive")
    
    if agg_factor <= 0:
      raise ValueError("agg_factor must be positive")

    fine_contexts = []
    coarse_contexts = []
    fine_pads = []
    coarse_pads = []
    offsets_fine = []
    scales_fine = []
    global_eps = 1e-8

    horizon_len = horizon_len or self.output_patch_len

    CONTEXT_LEN_FINE = 512
    CONTEXT_LEN_COARSE = 512

    inputs = self._normalize_inputs(inputs)

    for seq in inputs:
      series = np.array(seq)
      if not np.isfinite(series).all():
        series = np.where(np.isfinite(series), series, np.nan)
      series = strip_leading_nans(series)
      series = linear_interpolation(series)

      coarse_ctx, fine_ctx = self.build_multi_resolution(series, agg_factor=agg_factor)

      # Raw tensors
      ctx_coarse = torch.tensor(coarse_ctx,  dtype=torch.float32)
      ctx_fine = torch.tensor(fine_ctx, dtype=torch.float32)
      
      # Pad / truncate
      ctx_coarse_pad, mask_coarse  = self._pad_or_truncate(ctx_coarse,  CONTEXT_LEN_COARSE)
      ctx_fine_pad, mask_fine = self._pad_or_truncate(ctx_fine, CONTEXT_LEN_FINE)

      # Add batch dim
      ctx_coarse_pad_b = ctx_coarse_pad.unsqueeze(0)
      mask_coarse_b = mask_coarse.unsqueeze(0)
      ctx_fine_pad_b = ctx_fine_pad.unsqueeze(0)
      mask_fine_b = mask_fine.unsqueeze(0)

      # Normalize
      norm_coarse, _, _, _ = self.normalize_with_pad(ctx_coarse_pad_b, pad_mask=mask_coarse_b)
      norm_fine, offset_fine, scale_fine, _ = self.normalize_with_pad(ctx_fine_pad_b, pad_mask=mask_fine_b)

      # Store normalized contexts
      coarse_contexts.append(norm_coarse.squeeze(0).numpy())    # [L_coarse]
      coarse_pads.append(mask_coarse_b.squeeze(0).numpy())
      fine_contexts.append(norm_fine.squeeze(0).numpy())  # [L_fine]
      fine_pads.append(mask_fine_b.squeeze(0).numpy())

      # Store scalar stats only from fine contexts
      offsets_fine.append(float(offset_fine.squeeze()))
      scales_fine.append(float(scale_fine.squeeze()))

    # Arrays of shape [N]
    offsets_fine = np.array(offsets_fine, dtype=np.float32)
    scales_fine  = np.array(scales_fine, dtype=np.float32)

    N = len(fine_contexts)
    final_predictions = []

    for start in range(0, N, batch_size):
      end = min(start + batch_size, N)

      batch_coarse = torch.as_tensor(coarse_contexts[start:end],  dtype=torch.float32).unsqueeze(-1).to(self._device)
      batch_coarse_pad = torch.as_tensor(coarse_pads[start:end],  dtype=torch.float32).unsqueeze(-1).to(self._device)
      batch_fine = torch.as_tensor(fine_contexts[start:end], dtype=torch.float32).unsqueeze(-1).to(self._device)
      batch_fine_pad = torch.as_tensor(fine_pads[start:end], dtype=torch.float32).unsqueeze(-1).to(self._device)

      freq_tensor = torch.zeros((end - start, 1), dtype=torch.long, device=self._device)

      with torch.no_grad():
        preds = self._model.decode([batch_coarse, batch_fine],
                                   [batch_coarse_pad.float(), batch_fine_pad.float()],
                                   freq_tensor, 
                                   horizon_len=horizon_len, 
                                   agg_factor=agg_factor, 
                                   offsets=offsets_fine[start:end],
                                   scales=scales_fine[start:end], 
                                   global_eps=global_eps,
                                   output_patch_len=self.output_patch_len)

      final_predictions += preds

    return final_predictions
