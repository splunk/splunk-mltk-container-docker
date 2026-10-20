#!/usr/bin/env bash
# Build Splunk DSDL container images (docker buildx bake, configured by images.toml).
#
# Usage:
#   ./build.sh                                 list build targets and published names
#   ./build.sh <name> [repo] [version]         build one target and load it locally
#   ./build.sh all [repo] [version]            build every target
#   ./build.sh --push <name|all> [repo] [version]
#                                              build all declared platforms, push with
#                                              SBOM and provenance, then create the
#                                              single-platform aliases (golden-cpu:<v>-arm64)
#
# <name> is a target (cpu, llm, agentic, gpu, rapids, spark) or a 5.2.x name such as
# golden-cpu, golden-cpu-arm, escu-cpu, golden-gpu-transformers or ubi-llm-rag.
#
# Local builds load ONE platform: a 5.2.x name builds the platform it was published for
# (golden-cpu-arm: linux/arm64, the others linux/amd64); a target name builds the host's
# platform if the target supports it, else its first platform. PLATFORM=linux/arm64 picks
# a platform, PLATFORM=all loads every declared platform (needs the containerd image store).
# Building for a foreign platform uses QEMU emulation and is slow.
#
# Every build writes images_conf_files/images.conf with the stanzas of what was built.
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
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repo" -w /repo "$builder_image" \
      python3.12 tools/render.py "$@"
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
export DSDL_PYTHON_VERSION="$(sed -n 's/^python = "\(.*\)"/\1/p' images.toml)"
echo "Building ${targets[*]} as ${repo}*:${version} (revision ${VCS_REF}, Python ${DSDL_PYTHON_VERSION})"

# the multi-platform index reference of the same repository as a single-platform alias
index_digest() { # index-reference platform
  docker buildx imagetools inspect "$1" --raw | python3 -c '
import json, sys
os_, arch = sys.argv[1].split("/")
for m in json.load(sys.stdin).get("manifests", []):
    p = m.get("platform", {})
    if p.get("os") == os_ and p.get("architecture") == arch:
        print(m["digest"])
        break' "$2"
}

if [ "$push" = true ]; then
  docker buildx bake -f docker-bake.json --push \
    --set '*.attest=type=sbom' --set '*.attest=type=provenance,mode=max' "${targets[@]}"
  # Single-platform aliases point at one platform manifest of the pushed index. They carry
  # no attestations of their own: the SBOM and provenance stay attached to the index.
  render --aliases --repo "$repo" --version "$version" | while read -r target platform alias index; do
    case " ${targets[*]} " in *" $target "*) ;; *) continue ;; esac
    digest="$(index_digest "$index" "$platform")"
    if [ -z "$digest" ]; then
      echo "No $platform manifest in $index; cannot create $alias" >&2
      exit 1
    fi
    echo "Alias $alias -> ${index%:*}@$digest ($platform)"
    docker buildx imagetools create --prefer-index=false -t "$alias" "${index%:*}@$digest"
  done
  only="$(IFS=,; echo "${targets[*]}")"
else
  host_platform="linux/$(docker version --format '{{.Server.Arch}}' 2>/dev/null || uname -m)"
  alias_ref="$(render --alias-ref "$name" --repo "$repo" --version "$version")"
  for target in "${targets[@]}"; do
    declared="$(render --platforms "$target")"
    if [ -n "$alias_ref" ]; then
      platform="$(render --platforms "$name")"           # e.g. golden-cpu-arm -> linux/arm64
    elif [ "${PLATFORM:-}" = "all" ]; then
      platform="$declared"
    elif [ -n "${PLATFORM:-}" ]; then
      platform="$PLATFORM"
    elif [ "$name" = "$target" ] || [ "$name" = "all" ]; then
      case ",$declared," in *",$host_platform,"*) platform="$host_platform" ;; *) platform="${declared%%,*}" ;; esac
    else
      platform="$(render --platforms "$name")"           # 5.2.x name: its published platform
    fi
    case ",$declared," in
      *",${platform%%,*},"*) ;;
      *) if [ "$name" = "all" ]; then echo "Skipping $target: no $platform build"; continue; fi
         echo "$target is not built for $platform (supported: $declared)" >&2; exit 2 ;;
    esac
    if [ -n "$alias_ref" ]; then
      # a single-platform 5.2.x name: tag only that name, keep the multi-platform tags
      docker buildx bake -f docker-bake.json --load --set "$target.platform=$platform" \
        --set "$target.tags=$alias_ref" "$target"
    else
      docker buildx bake -f docker-bake.json --load --set "$target.platform=$platform" "$target"
      # local equivalent of the single-platform alias when exactly that platform was built
      render --aliases --repo "$repo" --version "$version" | while read -r alias_target alias_platform alias index; do
        if [ "$alias_target" = "$target" ] && [ "$platform" = "$alias_platform" ]; then
          docker tag "$index" "$alias"
        fi
      done
    fi
  done
  if [ "$name" = "all" ]; then only="$(IFS=,; echo "${targets[*]}")"
  elif [ "$name" = "${targets[0]}" ]; then only="$name"
  else only="$name"                                      # a 5.2.x name: just its stanza
  fi
fi

render --images-conf --repo "$repo" --version "$version" --only "$only"
echo "images_conf_files/images.conf holds the stanzas of this build ($only); copy them into"
echo "<splunk>/etc/apps/mltk-container/local/images.conf"
