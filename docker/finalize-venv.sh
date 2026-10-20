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
# bitsandbytes ships one native library per CUDA release (11.8-13.x), ROCm and XPU; keep only
# the build matching the installed torch CUDA version (about 150 MB and their scanner findings)
bnb="$site/bitsandbytes"
torch_cuda="$(sed -n "s/^cuda: Optional\[str\] = '\([0-9]*\)\.\([0-9]*\)'.*/\1\2/p; s/^cuda = '\([0-9]*\)\.\([0-9]*\)'.*/\1\2/p" "$site/torch/version.py" 2>/dev/null | head -1)"
if [ -d "$bnb" ] && [ -n "$torch_cuda" ] && [ -e "$bnb/libbitsandbytes_cuda${torch_cuda}.so" ]; then
  find "$bnb" -maxdepth 1 -name 'libbitsandbytes_*.so' ! -name "libbitsandbytes_cuda${torch_cuda}.so" \
    ! -name 'libbitsandbytes_cpu.so' -delete
fi
