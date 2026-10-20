# timesfm 1.3.0.post1 (vendored)

The TimesFM 1.x **PyTorch inference path**, vendored for the Splunk DSDL GPU image
(`deps/gpu-cu12`). It is the upstream 1.3.0 release with packaging changes only. The
Python sources are byte-identical to upstream.

## Origin

| | |
|---|---|
| Project | TimesFM, Google Research, <https://github.com/google-research/timesfm> |
| Release | `timesfm==1.3.0` on PyPI |
| Taken from | `timesfm-1.3.0.tar.gz`, sha256 `c494775c17d373d8d7f2e5562468fdff371c123d1d6e43dcfe92fbf9a86fec4e` (its `src/timesfm/` is identical to the `timesfm-1.3.0-py3-none-any.whl` wheel, sha256 `a99e22b13654bdbfaecb14a48e4a8dc6c990b191a0322a5cacd09eb0df3bfebe`) |
| License | Apache License 2.0. `LICENSE` is the upstream file, unchanged (sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`, identical to the `LICENSE` on the upstream `master` branch). Copyright 2024 Google LLC; every source file keeps its Apache-2.0 header |

## Why it is vendored

DSDL ships four modules that use the TimesFM 1.x API: `app/model/cisco_tsm_mr.py`,
`tsfm_forecast.py`, `timesfm_multi_resolution.py` and
`patched_decoder_multi_resolution.py`. They use `TimesFmHparams`, `TimesFmCheckpoint`,
`timesfm_torch.TimesFmTorch` (subclassed), `timesfm_base.strip_leading_nans` /
`linear_interpolation` and `pytorch_patched_decoder`.

The 1.3.0 release on PyPI cannot be installed into the DSDL images:

- It declares `Requires-Python: >=3.10,<3.12`. The images use Python 3.12. The upper
  bound only comes from the jax / paxml / lingvo extras, not from the PyTorch path.
- It depends on `wandb`. wandb ships a Go binary (`wandb-core`) with 120 scanner findings,
  54 of them High or Critical, in 5.2.5 golden-gpu-transformers. It also brings `typer`,
  and through the finetuning example `yfinance`, `curl-cffi`, `peewee` and
  `multitasking`.
- TimesFM 2.x and 3.x have a different API, so a newer release needs a code port first.

## Contents

`src/timesfm/` is the complete upstream package with no edits:

| File | sha256 (= upstream 1.3.0) |
|---|---|
| `__init__.py` | `a5d9234dd00365c33138a63cd4b60c88e05f8d6c614d497146de241b6f66e23c` |
| `data_loader.py` | `7a1a8ce7b100a8aaf19f0bf36816a31089ce81c6c8dc318831feea00a515f0c5` |
| `patched_decoder.py` | `fb5d9b1d4fe23510e708ac9249d026601625fb04ce0fe2336e26cf07607d349d` |
| `pytorch_patched_decoder.py` | `2b231c69a9d34bca89eedde8dbe6e77778b8d4f8837515dddd6fac9ef4759965` |
| `time_features.py` | `d904fc57c6814a07d2638718fbfdaa7f436a20cccbe365055b4a2da31f888d9d` |
| `timesfm_base.py` | `ac8b8ce3435b8e1a592cbc38a395b5a32787c8b75be3690eb76493d6674ebf64` |
| `timesfm_jax.py` | `84a55608a0278f8d97b005989de487b2147ccbc6ce3148c6c2aec14705e3c4a2` |
| `timesfm_torch.py` | `aa822d90772c39631dddbc6b93080bdfe769d322e2234925c24771675a69b089` |
| `xreg_lib.py` | `c12aac1fbb6b5df7aeb57b7a9c8a2cf5d8431f5574117baad53101fab5909074` |

To check the files: `cd vendor/timesfm && shasum -a 256 src/timesfm/*.py`.

## Modifications against upstream 1.3.0

All changes are to packaging. No Python source line was changed.

1. **`pyproject.toml` rewritten.** The Poetry metadata became PEP 621 metadata, built
   with `uv_build`. The DSDL profile pins the build backend to the uv release
   (`build-constraint-dependencies` in `deps/gpu-cu12/pyproject.toml`).
2. **Version `1.3.0.post1`.** The version marks the repackaging, so the dist cannot be
   mistaken for the PyPI artefact.
3. **`requires-python = ">=3.10"`.** The upstream `<3.12` bound is dropped. The
   PyTorch path has no 3.12 incompatibility. All nine modules byte-compile on 3.12, and
   the torch path runs on CPython 3.12.15 with torch 2.14.1.
4. **Dependencies reduced** to what the PyTorch path imports: `numpy`, `pandas`,
   `utilsforecast` (used by `timesfm_base`), `huggingface-hub` (`timesfm_torch`) and
   `torch`. Removed: `wandb`, `typer`, `absl-py`, `einshape`, `scikit-learn`, the
   `[cli]` extra of huggingface_hub, and the `pax` and `torch` extras (jax, jaxlib,
   paxml, lingvo).
5. **The `finetuning` top-level package is not vendored.** It contains
   `finetuning_torch.py` and `finetuning_example.py`, which import `wandb`, `yfinance`
   and `absl` at module level. DSDL does not use it.
6. The upstream `README.md` is replaced by this file.

## The jax backend

`timesfm_jax.py`, `patched_decoder.py`, `xreg_lib.py` and `data_loader.py` are kept
unmodified so that the package matches upstream. They import jax, praxis, paxml,
tensorflow or absl, and DSDL does not install jax, praxis or paxml. No patch is needed:

- Upstream `timesfm/__init__.py` already wraps the jax import in
  `try: ... except Exception:`.
- Without jax it binds `timesfm.TimesFm` to `TimesFmTorch`.

As a result, `import timesfm` works without jax. This was validated with CPython
3.12.15, torch 2.14.1 and the gpu-cu12 lock versions of numpy, pandas, huggingface-hub,
utilsforecast and datasets: `timesfm.TimesFm is TimesFmTorch`, and jax is never loaded.
Tiny random-weight `TimesFmTorch.forecast` and `CiscoTsmMR.forecast` runs also passed.

Consequences:

- `TimesFmBase.forecast_with_covariates` lazily imports `xreg_lib`, which needs jax, so
  it is unavailable. No DSDL module calls it.
- Upstream behaviour is kept: `import timesfm` prints two informational lines to stdout.

## Removal

Delete `vendor/timesfm/` and the `timesfm` entries in `deps/gpu-cu12/pyproject.toml`
once the four DSDL modules are ported to the TimesFM 2.x or 3.x API, which is on PyPI
and depends only on numpy.
