#!/usr/bin/env bash
# Re-pin the base images in docker/Dockerfile to their current digests.
#
# Usage: tools/update-digests.sh [uv-version]
#
# UBI 9 and ubi-micro are rebuilt by Red Hat several times a month; re-pinning and
# rebuilding is how OS fixes reach the DSDL images. Review the diff, rebuild, run
# tools/smoke-test.sh and tools/scan.sh --gate, then commit.
set -Eeuo pipefail

cd "$(dirname "$0")/.."
dockerfile=docker/Dockerfile
uv_version="${1:-$(sed -n 's/^ARG UV_IMAGE=ghcr.io\/astral-sh\/uv:\([^@]*\)@.*/\1/p' "$dockerfile")}"

pin() { # ARG name, image reference without digest
  local arg="$1" ref="$2" digest
  digest="$(docker buildx imagetools inspect "$ref" --format '{{json .Manifest.Digest}}' | tr -d '"')"
  sed -i.bak "s|^ARG $arg=.*|ARG $arg=${ref%:latest}@${digest}|" "$dockerfile" && rm -f "$dockerfile.bak"
  echo "$arg -> ${ref%:latest}@${digest}"
}

pin UBI_IMAGE registry.access.redhat.com/ubi9/ubi:latest
pin MICRO_IMAGE registry.access.redhat.com/ubi9/ubi-micro:latest
pin UV_IMAGE "ghcr.io/astral-sh/uv:${uv_version}"
git --no-pager diff --stat -- "$dockerfile" || true
