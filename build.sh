#!/usr/bin/env bash
# Build Splunk DSDL container images (docker buildx bake, configured by images.toml).
#
# Usage:
#   ./build.sh                                 list build targets and published names
#   ./build.sh <name> [repo] [version]         build one target and load it locally
#   ./build.sh all [repo] [version]            build every target
#   ./build.sh --push <name|all> [repo] [version]
#                                              build all platforms, push with SBOM and
#                                              provenance, create the single-platform
#                                              aliases (e.g. golden-cpu:<v>-arm64)
#
# <name> is a target (cpu, llm, agentic, gpu, rapids, spark) or a 5.2.x name such as
# golden-cpu, golden-cpu-arm, escu-cpu, golden-gpu-transformers or ubi-llm-rag.
# A 5.2.x name builds the platform it was published for (amd64, or arm64 for
# golden-cpu-arm); a target name builds all of its platforms. PLATFORM=linux/arm64
# overrides the platform of a local build.
#
# Every build writes images_conf_files/images.conf for the DSDL app.
set -Eeuo pipefail

cd "$(dirname "$0")"

push=false
if [ "${1:-}" = "--push" ]; then push=true; shift; fi
name="${1:-}"
repo="${2:-splunk/}"
version="${3:-$(sed -n 's/^version = "\(.*\)"/\1/p' images.toml)}"

builder_image="dsdl-builder:local"
render() {
  if python3 -c 'import sys; sys.version_info >= (3, 11) or __import__("tomli")' 2>/dev/null; then
    python3 tools/render.py "$@"
  else
    if ! docker image inspect "$builder_image" >/dev/null 2>&1; then
      docker buildx build --load --quiet --target builder -t "$builder_image" -f docker/Dockerfile . >/dev/null
    fi
    docker run --rm -v "$PWD:/repo" -w /repo "$builder_image" python3.12 tools/render.py "$@"
  fi
}

if [ -z "$name" ]; then
  echo "Build targets:"
  render --targets | sed 's/^/  /'
  echo "Published names (stanza -> image), see images.toml:"
  sed -n 's/.*stanza = "\([^"]*\)", image = "\([^"]*\)", tag = "\([^"]*\)".*/  \1 -> \2:\3/p' images.toml
  exit 0
fi

render --check || render
if [ "$name" = "all" ]; then
  targets=($(render --targets))
else
  targets=("$(render --resolve "$name")")
fi

export REPO="$repo" VERSION="$version" VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Building ${targets[*]} as ${repo}*:${version} (revision ${VCS_REF})"

if [ "$push" = true ]; then
  docker buildx bake -f docker-bake.json --push \
    --set '*.attest=type=sbom' --set '*.attest=type=provenance,mode=max' "${targets[@]}"
  # single-platform aliases: point <image>:<tag> at one platform manifest of the index
  render --aliases --repo "$repo" --version "$version" | while read -r target platform alias index; do
    case " ${targets[*]} " in *" $target "*) ;; *) continue ;; esac
    digest="$(docker buildx imagetools inspect "$index" --raw | python3 -c '
import json, sys
os_, arch = sys.argv[1].split("/")
for m in json.load(sys.stdin)["manifests"]:
    p = m.get("platform", {})
    if p.get("os") == os_ and p.get("architecture") == arch:
        print(m["digest"]); break' "$platform")"
    echo "Alias $alias -> ${index%:*}@$digest ($platform)"
    docker buildx imagetools create --prefer-index=false -t "$alias" "${index%:*}@$digest"
  done
else
  for target in "${targets[@]}"; do
    # 5.2.x names keep their published platform; target names build all platforms
    if [ "$name" = "all" ]; then
      platform="${PLATFORM:-$(render --platforms "$target")}"
    else
      platform="${PLATFORM:-$(render --platforms "$name")}"
    fi
    docker buildx bake -f docker-bake.json --load --set "$target.platform=$platform" "$target"
    # local equivalents of the single-platform aliases
    render --aliases --repo "$repo" --version "$version" | while read -r alias_target alias_platform alias index; do
      if [ "$alias_target" = "$target" ] && [[ ",$platform," == *",$alias_platform,"* ]]; then
        docker tag "$index" "$alias"
      fi
    done
  done
fi

render --images-conf --repo "$repo" --version "$version"
echo "Copy images_conf_files/images.conf into <splunk>/etc/apps/mltk-container/local/images.conf"
