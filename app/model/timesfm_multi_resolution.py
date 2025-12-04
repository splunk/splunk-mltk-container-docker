import logging
from os import path
from typing import Any, List, Sequence, Union

import numpy as np
import torch

from huggingface_hub import snapshot_download

from timesfm import TimesFmHparams, TimesFmCheckpoint
from timesfm.timesfm_torch import TimesFmTorch
from timesfm.timesfm_base import strip_leading_nans, linear_interpolation

from .patched_decoder_multi_resolution import TimesfmMRConfig, PatchedTSMultiResolutionDecoder


class TimesFmMRTorch(TimesFmTorch):
  """TimesFM PyTorch wrapper using the MR-extended decoder/config.

  This class keeps the same forecast API as TimesFmTorch, while
  allowing toggle between multi-resolution behaviors via init kwargs.
  """

  def __init__(
      self,
      hparams: TimesFmHparams,
      checkpoint: TimesFmCheckpoint,
      *,
      use_multi_resolution: bool = False,
      use_special_token_s: bool = False,
      use_multi_task: bool = False,
      horizon_len_hr: int | None = None,
  ) -> None:
    self.use_multi_resolution = use_multi_resolution
    self.use_special_token_s = use_special_token_s
    self.use_multi_task = use_multi_task
    self.horizon_len_hr = horizon_len_hr
    super().__init__(hparams, checkpoint)

  def __post_init__(self):
    # Build MR config from upstream hparams + MR flags
    self._model_config = TimesfmMRConfig(
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        hidden_size=self.model_dims,
        intermediate_size=self.model_dims,
        patch_len=self.input_patch_len,
        horizon_len=self.output_patch_len,
        head_dim=self.model_dims // self.num_heads,
        quantiles=self.quantiles,
        use_positional_embedding=self.use_pos_emb,
        use_multi_resolution=self.use_multi_resolution,
        use_special_token_s=self.use_special_token_s,
        use_multi_task=self.use_multi_task,
        horizon_len_hr=(self.horizon_len_hr if self.horizon_len_hr is not None else self.output_patch_len),
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
    """Loads a checkpoint from path"""

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


  def _pad_or_truncate(self, ts: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate a time series to a target length. [LEFT-PADDING]"""
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

  
  def _normalize(self, context, pad_mask, normalize_method="standard"):
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

    if normalize_method == "standard":
        ctx_normalized = (context - context_mean) / (context_std + eps)
        stats = (context_mean, context_std)

    elif normalize_method == "minmax":
        masked_context = context.clone()
        masked_context[pad_mask == 1] = float("inf")
        cmin = masked_context.min(dim=1, keepdim=True).values
        cmin[~torch.isfinite(cmin)] = context_mean[~torch.isfinite(cmin)]

        masked_context = context.clone()
        masked_context[pad_mask == 1] = float("-inf")
        cmax = masked_context.max(dim=1, keepdim=True).values
        cmax[~torch.isfinite(cmax)] = context_mean[~torch.isfinite(cmax)]

        ctx_normalized = (context - cmin) / (cmax - cmin + eps)
        stats = (cmin, cmax - cmin)

    elif normalize_method == "robust_zscore":
        masked_context = context.clone()
        masked_context[pad_mask == 1] = context_mean.expand_as(masked_context)[pad_mask == 1]
        
        context_median = masked_context.median(dim=1, keepdim=True).values

        # mean adjusted deviation
        mad = (masked_context - context_median).abs().median(dim=1, keepdim=True).values
        scaled_mad = 1.4826 * mad

        ctx_normalized = (context - context_median) / (scaled_mad + eps)
        stats = (context_median, scaled_mad)

    elif normalize_method == "iqr":
        masked_context = context.clone()
        masked_context[pad_mask == 1] = context_mean.expand_as(masked_context)[pad_mask == 1]
        
        q1 = masked_context.quantile(0.25, dim=1, keepdim=True)
        q3 = masked_context.quantile(0.75, dim=1, keepdim=True)
        
        context_iqr = q3 - q1
        
        context_median = masked_context.median(dim=1, keepdim=True).values
        ctx_normalized = (context - context_median) / (context_iqr + eps)
        stats = (context_median, context_iqr)
    
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")

    ctx_normalized = ctx_normalized * valid

    # Clamp to a safe range to avoid numerical issues
    ctx_normalized = torch.clamp(ctx_normalized, -1000.0, 1000.0)

    offset, scale = stats

    return ctx_normalized, offset, scale, eps

  
  def forecast(self, inputs: Sequence[Any], agg_factor: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts from a single high-resolution stream.

    Derives the low-resolution stream by aggregating the high-resolution
    context in blocks of `agg_factor` (e.g., 60 minutes -> hourly) and then
    runs multi-resolution decoding.

    Args:
      inputs: list-like of high-resolution context series.
      agg_factor: size of aggregation window to form the low-resolution
        context from the high-resolution context.

    Returns:
      Tuple (mean_forecast, full_forecast).
    """
    if self._model is None:
      raise ValueError("Checkpoint is not properly loaded.")

    # Both high and low-resolution context lengths to context_len
    forecast_context_len = self.context_len
    low_context_len = self.context_len

    # Normalize and pad/truncate each series individually (robust to length mismatches)
    norm_high_list: list[np.ndarray] = []
    norm_low_list: list[np.ndarray] = []
    pad_high_list: list[np.ndarray] = []
    pad_low_list: list[np.ndarray] = []
    offsets_high: list[float] = []
    scales_high: list[float] = []

    # Clean inputs (full length) and build normalized contexts
    for seq in inputs:
      arr = np.array(seq)
      if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, np.nan)
      arr = strip_leading_nans(arr)
      arr = linear_interpolation(arr)
      arr = arr.astype(np.float32, copy=False)

      # High-res: take last `forecast_context_len`, pad/truncate, then normalize with mask
      ctx_high = arr[-forecast_context_len:]
      t_ctx_high = torch.tensor(ctx_high, dtype=torch.float32)
      t_ctx_high_pad, mask_high = self._pad_or_truncate(t_ctx_high, forecast_context_len)
      t_ctx_high_pad_b = t_ctx_high_pad.unsqueeze(0)
      mask_high_b = mask_high.unsqueeze(0)
      norm_high, off_h, sc_h, _ = self._normalize(t_ctx_high_pad_b, mask_high_b)
      norm_high_list.append(norm_high.squeeze(0).detach().cpu().numpy())
      pad_high_list.append(mask_high.detach().cpu().numpy())
      offsets_high.append(float(off_h.squeeze().detach().cpu().numpy()))
      scales_high.append(float(sc_h.squeeze().detach().cpu().numpy()))

      # Low-res: derive from full series by backward aggregation; pad/truncate to a fixed length
      L = arr.shape[0]
      usable = (L // agg_factor) * agg_factor
      if usable <= 0:
        low_series = np.array([], dtype=np.float32)
      else:
        blocks = arr[-usable:].reshape(-1, agg_factor)
        low_series = blocks.mean(axis=1).astype(np.float32)
      
      t_low = torch.tensor(low_series, dtype=torch.float32)
      t_low_pad, mask_low = self._pad_or_truncate(t_low, low_context_len)
      t_low_pad_b = t_low_pad.unsqueeze(0)
      mask_low_b = mask_low.unsqueeze(0)
      norm_low, _, _, _ = self._normalize(t_low_pad_b, mask_low_b)
      norm_low_list.append(norm_low.squeeze(0).detach().cpu().numpy())
      pad_low_list.append(mask_low.detach().cpu().numpy())

    # Build dense tensors for _forecast (uniform shapes)
    inputs_high_t = torch.tensor(np.stack(norm_high_list, axis=0), dtype=torch.float32)
    inputs_low_t = torch.tensor(np.stack(norm_low_list, axis=0), dtype=torch.float32)

    # Frequency list (0=high by default)
    freq = [0] * inputs_high_t.shape[0]

    # If horizon equals output_patch_len, mirror notebook: call model forward directly with true masks.
    if self.horizon_len == self.output_patch_len:
      B = inputs_high_t.shape[0]
      device = self._device
      with torch.no_grad():
        t_low = inputs_low_t.unsqueeze(-1).to(device)
        t_high = inputs_high_t.unsqueeze(-1).to(device)
        p_low = torch.tensor(np.stack(pad_low_list, axis=0), dtype=torch.float32, device=device).unsqueeze(-1)
        p_high = torch.tensor(np.stack(pad_high_list, axis=0), dtype=torch.float32, device=device).unsqueeze(-1)
        t_freq = torch.zeros((B, 1), dtype=torch.long, device=device)

        preds = self._model([t_low, t_high], [p_low, p_high], t_freq)
        last_patch = preds[:, -1, :self.horizon_len, :]
        mean_forecast = last_patch[..., 0].detach().cpu().numpy()
        quantile_forecast = last_patch.detach().cpu().numpy()
    else:
      mean_forecast, quantile_forecast = self._forecast(
          inputs_low_t,
          inputs_high_t,
          freq=freq,
          agg_factor=agg_factor,
      )

    # Denormalize forecasts to original high-res scale
    _scale_2d = np.array(scales_high, dtype=np.float32).reshape(-1, 1)
    _offset_2d = np.array(offsets_high, dtype=np.float32).reshape(-1, 1)
    mean_forecast = mean_forecast * _scale_2d + _offset_2d
    _scale_3d = _scale_2d.reshape(-1, 1, 1)
    _offset_3d = _offset_2d.reshape(-1, 1, 1)
    quantile_forecast = quantile_forecast * _scale_3d + _offset_3d

    return mean_forecast, quantile_forecast


  def _forecast(
      self,
      inputs_low: Sequence[Any],
      inputs_high: Sequence[Any],
      freq: Sequence[int] | None = None,
      agg_factor: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts using two streams (low/high) with MR decoder.

    This method prepares inputs for the decoder by padding/truncating to the
    model context length, batching, and invoking the multi-resolution decode.
    It also passes `agg_factor` to the decoder so each autoregressive step uses
    only the first 120 minute points for the high-resolution progression and
    aggregates them into low-resolution hour points (size `agg_factor`).

    Args:
      inputs_low: sequence of per-item low-resolution contexts. Accepts either
        a ragged list of 1D numpy arrays (recommended) or a dense torch.Tensor;
        ragged inputs are padded internally.
      inputs_high: sequence of per-item high-resolution contexts. Accepts a
        ragged list of 1D numpy arrays or a dense torch.Tensor.
      freq: list of integer frequency codes per item (0=high by default).
      agg_factor: aggregation factor used inside decode for minute→hour
        aggregation feeding the low-resolution stream.

    Returns:
      A tuple (mean_forecast, full_forecast):
        - mean_forecast: numpy array shaped (#items, horizon_len)
        - full_forecast: numpy array shaped (#items, horizon_len, 1+Q)
    """

    # Preprocess both streams
    # Ensure numpy arrays of shape [L] per series as expected by _preprocess
    if isinstance(inputs_low, torch.Tensor):
      inputs_low = [inputs_low[i].detach().cpu().numpy() for i in range(inputs_low.shape[0])]
    if isinstance(inputs_high, torch.Tensor):
      inputs_high = [inputs_high[i].detach().cpu().numpy() for i in range(inputs_high.shape[0])]

    input_ts_l, input_pad_l, inp_freq_l, pmap_pad_l = self._preprocess(inputs_low, freq)
    input_ts_h, input_pad_h, inp_freq_h, pmap_pad_h = self._preprocess(inputs_high, freq)

    if pmap_pad_l != pmap_pad_h:
      logging.warning("pmap padding differs between streams: low=%d high=%d; proceeding with high.", pmap_pad_l, pmap_pad_h)
    pmap_pad = max(pmap_pad_l, pmap_pad_h)

    with torch.no_grad():
      mean_outputs = []
      full_outputs = []
      Btot = input_ts_h.shape[0]
      G = self.global_batch_size
      
      for i in range(Btot // G):
        sl = slice(i * G, (i + 1) * G)
        t_low = torch.tensor(input_ts_l[sl], dtype=torch.float32).to(self._device)
        t_high = torch.tensor(input_ts_h[sl], dtype=torch.float32).to(self._device)
        p_low = torch.tensor(input_pad_l[sl], dtype=torch.float32).to(self._device)
        p_high = torch.tensor(input_pad_h[sl], dtype=torch.float32).to(self._device)
        t_freq = torch.LongTensor(inp_freq_h[sl]).to(self._device)

        # If single-step horizon, align with the direct forward-based inference
        if self.horizon_len == self.output_patch_len:
          p_low_ctx = p_low[:, :t_low.shape[1]]
          p_high_ctx = p_high[:, :t_high.shape[1]]
          preds = self._model([t_low, t_high], [p_low_ctx, p_high_ctx], t_freq)
          last_patch = preds[:, -1, :self.horizon_len, :]
          mean_output = last_patch[:, :, 0]
          full_output = last_patch
        else:
          mean_output, full_output = self._model.decode(
              input_ts=[t_low, t_high],
              paddings=[p_low, p_high],
              freq=t_freq,
              horizon_len=self.horizon_len,
              output_patch_len=self.output_patch_len,
              agg_factor=agg_factor,
          )

        if self.backend == "gpu":
          mean_output = mean_output.cpu()
          full_output = full_output.cpu()
        mean_output = mean_output.detach().numpy()
        full_output = full_output.detach().numpy()
        mean_outputs.append(mean_output)
        full_outputs.append(full_output)

    mean_outputs = np.concatenate(mean_outputs, axis=0)
    full_outputs = np.concatenate(full_outputs, axis=0)

    if pmap_pad > 0:
      mean_outputs = mean_outputs[:-pmap_pad, ...]
      full_outputs = full_outputs[:-pmap_pad, ...]

    return mean_outputs, full_outputs
  
