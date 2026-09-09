#!/usr/bin/env bash

set -Eeuo pipefail

readonly VERSION="5.2.5"
readonly REPOSITORY="splunk"

usage() {
  printf 'Usage: %s [--dry-run]\n' "$(basename "$0")"
  printf '\nPushes the ten publishable DSDL %s images to Docker Hub.\n' "$VERSION"
  printf 'The agentic-ai-debian test image is intentionally excluded.\n'
}

dry_run=false
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=true ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [ "$#" -gt 1 ]; then
  usage >&2
  exit 2
fi

# Keep this list aligned with images_conf_files/images.conf. Variants use the
# established 5.2.4 repository structure instead of their internal build names.
readonly images=(
  "${REPOSITORY}/mltk-container-golden-cpu:${VERSION}"
  "${REPOSITORY}/mltk-container-golden-cpu:${VERSION}-arm64"
  "${REPOSITORY}/mltk-container-golden-gpu:${VERSION}"
  "${REPOSITORY}/mltk-container-golden-cpu-transformers:${VERSION}"
  "${REPOSITORY}/mltk-container-golden-gpu-transformers:${VERSION}"
  "${REPOSITORY}/mltk-container-golden-gpu-rapids:${VERSION}"
  "${REPOSITORY}/mltk-container-ubi-llm-rag:${VERSION}"
  "${REPOSITORY}/mltk-container-escu-cpu:${VERSION}"
  "${REPOSITORY}/mltk-container-spark:${VERSION}"
  "${REPOSITORY}/mltk-container-ubi-llm-rag:agentic-ai-${VERSION}"
)

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not available on PATH." >&2
  exit 1
fi

echo "Checking all ${#images[@]} publishable images before pushing..."
missing=()
for image in "${images[@]}"; do
  # This guard ensures a future edit cannot accidentally publish the test image.
  if [[ "$image" == *agentic-ai-debian* ]]; then
    echo "Refusing to publish excluded test image: $image" >&2
    exit 1
  fi

  if ! docker image inspect "$image" >/dev/null 2>&1; then
    missing+=("$image")
  fi
done

if [ "${#missing[@]}" -ne 0 ]; then
  echo "The following images are missing locally; nothing was pushed:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

if [ "$dry_run" = true ]; then
  echo "Dry run: all images exist locally. Images that would be pushed:"
  printf '  %s\n' "${images[@]}"
  exit 0
fi

echo "Pushing ${#images[@]} images to Docker Hub..."
pushed=0
for image in "${images[@]}"; do
  echo "Pushing $image"
  docker push "$image"
  pushed=$((pushed + 1))
done

echo "Successfully pushed $pushed images."
