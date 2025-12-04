import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm, trange
from typing import List, Tuple
import torch
from datasets import Dataset as HFDataset


MIN_CONTEXT_LEN_MINUTES = 512          # rightmost minute points to keep
MAX_HOURLY_CONTEXT = 512               # max hourly points (512 hours)
AGG_BLOCK = 60                         # 60 minutes -> 1 hour
CONTEXT_COL = "context_1min"               # column containing list of datapoints
PREDICTION_LENGTH = 128 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def slice_minute_context(series: List[float], minute_len: int = MIN_CONTEXT_LEN_MINUTES) -> List[float]:
    """Return the rightmost minute_len points (or entire series if shorter)."""
    if len(series) <= minute_len:
        return series[-minute_len:]  # may be shorter
    return series[-minute_len:]


def build_hourly_context(series: List[float], max_hours: int = MAX_HOURLY_CONTEXT, block: int = AGG_BLOCK) -> List[float]:
    """Construct hourly context by:
       1. Taking up to rightmost (max_hours * block) raw minute samples.
       2. Partitioning into consecutive non-overlapping blocks of 'block' size from left to right (chronological order preserved).
       3. Computing the mean of each block.
    Returns list of hourly means with length <= max_hours.
    """
    needed_raw = max_hours * block
    raw_slice = series[-needed_raw:]
    # Ensure we only form full blocks; drop partial leading block if length not multiple
    remainder = len(raw_slice) % block
    if remainder != 0:
        raw_slice = raw_slice[remainder:]  # align to block boundary at the right edge
    hourly = []
    for i in range(0, len(raw_slice), block):
        block_vals = raw_slice[i:i+block]
        if len(block_vals) < block:
            break
        hourly.append(float(sum(block_vals) / block))
    return hourly[-max_hours:]


def build_multi_resolution(series: List[float]) -> Tuple[List[float], List[float]]:
    minute_ctx = slice_minute_context(series)
    hourly_ctx = build_hourly_context(series)
    return minute_ctx, hourly_ctx

def process_input(series_list) -> List[dict]:
    '''
    series_list: [[1,2,3,4...,N]]
    '''
    outputs = []
    for idx, series in enumerate(series_list):
        if not series:
            continue
        context_1min, context_1hr = build_multi_resolution(series)
        outputs.append({
            "row_index": idx,
            "context_1min": context_1min,
            "context_1hr": context_1hr,
            "len_raw": len(series),
            "len_min": len(context_1min),
            "len_hr": len(context_1hr)
        })
    return outputs
