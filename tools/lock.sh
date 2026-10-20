#!/usr/bin/env bash
# Resolve, check or upgrade the hashed uv locks of the dependency profiles.
#
# Usage:
#   tools/lock.sh                 # resolve all profiles (keeps existing pins where possible)
#   tools/lock.sh cpu llm         # resolve selected profiles
#   tools/lock.sh --check         # fail if any uv.lock is out of date with its pyproject.toml
#   tools/lock.sh --upgrade cpu   # move every package to the newest allowed version
#
# The resolver runs inside the Dockerfile "builder" stage, i.e. against the same
# UBI 9 CPython that the images ship, so sdist metadata is built consistently.
set -Eeuo pipefail

cd "$(dirname "$0")/.."

mode=()
profiles=()
for arg in "$@"; do
  case "$arg" in
    --check) mode=(--check) ;;
    --upgrade) mode=(--upgrade) ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) profiles+=("$arg") ;;
  esac
done
if [ "${#profiles[@]}" -eq 0 ]; then
  profiles=()
  for dir in deps/*/; do profiles+=("$(basename "$dir")"); done
fi

builder_tag="dsdl-builder:lock"
docker buildx build --load --quiet --target builder -t "$builder_tag" -f docker/Dockerfile . >/dev/null

for profile in "${profiles[@]}"; do
  if [ ! -f "deps/$profile/pyproject.toml" ]; then
    echo "Unknown profile: $profile" >&2
    exit 2
  fi
  echo "uv lock ${mode[*]+${mode[*]}} -> deps/$profile/uv.lock"
  # the repo is mounted read-only (path sources such as ../../package-dsdlsupport);
  # only the profile directory is writable so uv can write uv.lock
  docker run --rm \
    -v "$PWD:/repo:ro" -v "$PWD/deps/$profile:/repo/deps/$profile" -w "/repo/deps/$profile" \
    -v dsdl-uv-cache:/root/.cache/uv \
    "$builder_tag" uv lock ${mode[@]+"${mode[@]}"}
done
