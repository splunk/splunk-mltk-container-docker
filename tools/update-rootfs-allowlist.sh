#!/usr/bin/env bash
# Rebuild the runtime rootfs without the allow-list check and write the resulting
# RPM list to docker/rootfs/allow.txt. Review the git diff before committing it.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
platform="${1:-linux/amd64}"
tag="dsdl-rootfs:allowlist"
touch docker/rootfs/allow.txt
docker buildx build --load --platform "$platform" --target rootfs --build-arg ROOTFS_ALLOWLIST_CHECK=0 \
  -t "$tag" -f docker/Dockerfile .
docker run --rm --platform "$platform" --entrypoint cat "$tag" /rootfs-packages.txt > docker/rootfs/allow.txt
echo "Wrote docker/rootfs/allow.txt ($(wc -l < docker/rootfs/allow.txt) packages)"
