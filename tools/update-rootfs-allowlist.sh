#!/usr/bin/env bash
# Rebuild the runtime rootfs (and the JRE delta of the Spark image) without the
# allow-list checks and write the resulting RPM lists to docker/rootfs/allow.txt and
# docker/rootfs/allow-jre.txt. Review the git diff before committing them.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
platform="${1:-linux/amd64}"
tag="dsdl-rootfs:allowlist"
touch docker/rootfs/allow.txt docker/rootfs/allow-jre.txt
docker buildx build --load --platform "$platform" --target rootfs-jre --build-arg ROOTFS_ALLOWLIST_CHECK=0 \
  -t "$tag" -f docker/Dockerfile .
docker run --rm --platform "$platform" --entrypoint cat "$tag" /rootfs-packages.txt > docker/rootfs/allow.txt
docker run --rm --platform "$platform" --entrypoint cat "$tag" /rootfs-jre-packages.txt > docker/rootfs/allow-jre.txt
docker image rm "$tag" >/dev/null
echo "Wrote docker/rootfs/allow.txt ($(wc -l < docker/rootfs/allow.txt) packages) and allow-jre.txt ($(wc -l < docker/rootfs/allow-jre.txt))"
