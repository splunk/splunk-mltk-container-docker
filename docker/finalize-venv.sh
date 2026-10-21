#!/bin/bash
# Final touches on a freshly synced /opt/dltk/venv (runs in the builder stage).
set -Eeuo pipefail

venv=/opt/dltk/venv
site="$(echo "$venv"/lib/python3.*/site-packages)"

# uv's lock file is world-writable and only meaningful while syncing
rm -f "$venv/.lock"
# tornado's test suite ships a private test key that secret scanners flag
rm -rf "$site/tornado/test"
# DEV-mode `pip install` goes to PIP_TARGET on the persistent /srv volume. The path is
# appended AFTER site-packages, so user packages can add to but never shadow image packages.
echo "/srv/.dsdl/site-packages" > "$site/zz-dsdl-user-packages.pth"
# TensorFlow loads its libraries with RTLD_GLOBAL; both it and triton bundle LLVM, and
# `import triton` after `import tensorflow` crashes the process (segfault, triton 3.7 and 3.8).
# bitsandbytes, peft and torch.compile import triton lazily, so a TensorFlow model followed by
# a quantised or compiled torch model in one process (DSDL API, Jupyter kernel) would crash.
# When both are installed (gpu image), triton is imported just before the first tensorflow
# import (about 0.3 s); the other order is safe.
if [ -d "$site/tensorflow" ] && [ -d "$site/triton" ]; then
  cat > "$site/_dsdl_triton_first.py" <<'EOF'
"""Import triton before tensorflow (written by DSDL finalize-venv; see there for why)."""
import sys


class TritonFirst:
    done = False

    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name == "tensorflow" and not cls.done:
            cls.done = True
            try:
                import triton  # noqa: F401
            except Exception:  # never block the tensorflow import
                pass
        return None


sys.meta_path.insert(0, TritonFirst)
EOF
  echo "import _dsdl_triton_first" > "$site/zz-dsdl-triton-first.pth"
fi
# bitsandbytes ships native libraries for CUDA 11.8, 12.1, 12.4, 12.6, 12.8, 13.0 and 13.2,
# six ROCm and two XPU releases; keep only the CPU build and the CUDA build it loads for the
# installed torch CUDA version (removes about 100 MB and their scanner findings). Same
# selection as bitsandbytes: the exact version, else the newest lower one of the same major
# (torch cu129 loads the cuda128 build), else the oldest higher one of the same major.
# BNB_CUDA_VERSION can therefore only name the kept build.
bnb="$site/bitsandbytes"
torch_cuda="$(sed -n "s/^cuda: Optional\[str\] = '\([0-9]*\.[0-9]*\)'.*/\1/p; s/^cuda = '\([0-9]*\.[0-9]*\)'.*/\1/p" "$site/torch/version.py" 2>/dev/null | head -1)"
if [ -d "$bnb" ] && [ -n "$torch_cuda" ]; then
  keep="$(ls "$bnb" | sed -n 's/^libbitsandbytes_cuda\([0-9][0-9]\)\([0-9]*\)\.so$/\1 \2/p' | awk -v want="$torch_cuda" '
    BEGIN { split(want, t, ".") }
    $1 == t[1] {
      if ($2 == t[2]) exact = $1 $2
      else if ($2 < t[2] && (lower == "" || $2 > lower_minor)) { lower = $1 $2; lower_minor = $2 }
      else if ($2 > t[2] && (higher == "" || $2 < higher_minor)) { higher = $1 $2; higher_minor = $2 }
    }
    END { print (exact != "" ? exact : (lower != "" ? lower : higher)) }')"
  if [ -n "$keep" ]; then
    find "$bnb" -maxdepth 1 -name 'libbitsandbytes_*.so' ! -name "libbitsandbytes_cuda${keep}.so" \
      ! -name 'libbitsandbytes_cpu.so' -delete
  fi
fi
