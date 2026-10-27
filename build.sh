#!/usr/bin/env bash
# Build Splunk DSDL container images (docker buildx bake, configured by images.toml).
#
# Usage:
#   ./build.sh                                 list build targets and published names
#   ./build.sh <name> [repo] [version]         build one target and load it locally
#   ./build.sh all [repo] [version]            build every target
#   ./build.sh --push-rebuild <name|all> [repo] [version]   (or --push)
#                                              build all declared platforms, push with
#                                              SBOM and provenance, then create the
#                                              single-platform aliases (golden-cpu:<v>-arm64)
#   ./build.sh --push-only [--dry-run] <name|all> [repo] [version]
#                                              push the images already built and loaded
#                                              here (PLATFORM=all), as tested, without a
#                                              rebuild and without SBOM or provenance (a
#                                              local load keeps none), then create the
#                                              aliases; --dry-run only checks and prints
#
# <name> is a target (cpu, llm, agentic, gpu, rapids) or a 5.2.x name such as
# golden-cpu, golden-cpu-arm, escu-cpu, spark, golden-gpu-transformers or ubi-llm-rag.
#
# Local builds load ONE platform: a 5.2.x name builds the platform it was published for
# (golden-cpu-arm: linux/arm64, the others linux/amd64); a target name builds the host's
# platform if the target supports it, else its first platform. PLATFORM=linux/arm64 picks
# a platform, PLATFORM=all loads every declared platform (needs the containerd image store).
# Building for a foreign platform uses QEMU emulation and is slow.
#
# Every build writes images_conf_files/images.conf with the stanzas of what was built.
#
# --push-only first checks every published tag of the targets: it must exist locally with
# all declared platforms and the version label <version>, and all tags of a target must be
# the same local image; otherwise nothing is pushed.
set -Eeuo pipefail

cd "$(dirname "$0")"

mode=build
case "${1:-}" in
  --push|--push-rebuild) mode=push-rebuild; shift ;;
  --push-only) mode=push-only; shift ;;
esac
dry_run=false
if [ "${1:-}" = "--dry-run" ]; then
  if [ "$mode" != push-only ]; then echo "--dry-run works with --push-only" >&2; exit 2; fi
  dry_run=true; shift
fi
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
case "$mode" in
  push-only) echo "Pushing the local ${targets[*]} images as ${repo}*:${version} (no rebuild$([ "$dry_run" = true ] && echo ", dry run"))" ;;
  *) echo "Building ${targets[*]} as ${repo}*:${version} (revision ${VCS_REF}, Python ${DSDL_PYTHON_VERSION})" ;;
esac

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

# Single-platform aliases point at one platform manifest of the pushed index. They carry
# no attestations of their own: SBOM and provenance (push-rebuild) stay attached to the index.
create_aliases() {
  render --aliases --repo "$repo" --version "$version" | while read -r target platform alias index; do
    case " ${targets[*]} " in *" $target "*) ;; *) continue ;; esac
    if [ "$dry_run" = true ]; then
      echo "would run: docker buildx imagetools create --prefer-index=false -t $alias ${index%:*}@<$platform manifest of the pushed $index>"
      continue
    fi
    digest="$(index_digest "$index" "$platform")"
    if [ -z "$digest" ]; then
      echo "No $platform manifest in $index; cannot create $alias" >&2
      exit 1
    fi
    echo "Alias $alias -> ${index%:*}@$digest ($platform)"
    docker buildx imagetools create --prefer-index=false -t "$alias" "${index%:*}@$digest"
  done
}

if [ "$mode" = push-rebuild ]; then
  docker buildx bake -f docker-bake.json --push \
    --set '*.attest=type=sbom' --set '*.attest=type=provenance,mode=max' "${targets[@]}"
  create_aliases
  only="$(IFS=,; echo "${targets[*]}")"
elif [ "$mode" = push-only ]; then
  # "<target> <platforms> <tag>" for every multi-platform tag, as the build resolves them
  plan="$(docker buildx bake -f docker-bake.json --progress=quiet --print "${targets[@]}" | python3 -c '
import json, sys
for name, target in json.load(sys.stdin)["target"].items():
    for tag in target.get("tags", []):
        print(name, ",".join(target.get("platforms", [])), tag)')"
  if [ -z "$plan" ]; then echo "No tags found for ${targets[*]}" >&2; exit 1; fi
  problems=0 first_target="" first_id="" first_tag=""
  while read -r target platforms tag; do
    # all tags of a target must be one local image: a tag left from an earlier build (e.g. of
    # a name that used to be its own target, like spark) would otherwise be pushed as it is
    id="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null || true)"
    if [ "$target" != "$first_target" ]; then
      first_target="$target" first_id="$id" first_tag="$tag"
    elif [ -n "$id" ] && [ -n "$first_id" ] && [ "$id" != "$first_id" ]; then
      echo "different image: $tag is not the image of $first_tag (both are tags of $target)" >&2; problems=1
    fi
    for platform in ${platforms//,/ }; do
      if ! labels="$(docker image inspect --platform "$platform" --format \
          '{{index .Config.Labels "org.opencontainers.image.version"}} {{index .Config.Labels "org.opencontainers.image.revision"}}' \
          "$tag" 2>/dev/null)"; then
        echo "missing locally: $tag ($platform)" >&2; problems=1
      elif [ "${labels%% *}" != "$version" ]; then
        echo "wrong version: $tag ($platform) is labelled '${labels%% *}', not $version" >&2; problems=1
      else
        echo "ready: $tag ($platform, revision ${labels#* })"
      fi
    done
  done <<< "$plan"
  if [ "$problems" != 0 ]; then
    echo "Nothing pushed. Build these images for every platform first: PLATFORM=all ./build.sh $name $repo $version" >&2
    exit 1
  fi
  while read -r target platforms tag; do
    if [ "$dry_run" = true ]; then echo "would run: docker push $tag   ($platforms)"; continue; fi
    echo "Pushing $tag ($platforms)"
    docker push --quiet "$tag"   # the image index with every platform loaded here
  done <<< "$plan"
  create_aliases
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
