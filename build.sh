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
#                                              aliases; --dry-run only checks and prints.
#                                              Prints a push plan (layers per tag, new data),
#                                              docker's per-layer progress, and the overall
#                                              percentage, elapsed time and time left
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
  # "<tag> <layers> <new layers> <new bytes>" per tag: its layers on all platforms, and those no
  # earlier tag of this push has with their uncompressed size, which weights the overall progress
  # (a tag of an image pushed before in this run only adds its manifests)
  weights="$(python3 - "$plan" <<'EOF'
import json, subprocess, sys
def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout
seen = set()
for line in sys.argv[1].splitlines():
    target, platforms, tag = line.split()
    layers = {}
    for platform in platforms.split(","):
        ids = json.loads(docker("image", "inspect", "--platform", platform, "--format", "{{json .RootFS.Layers}}", tag))
        # docker history lists the newest step first; its non-empty steps are the layers in order
        steps = [int(size) for size in reversed(docker("history", "--platform", platform, "--no-trunc",
                                                       "--human=false", "--format", "{{.Size}}", tag).split())]
        sizes = [size for size in steps if size > 0]
        if len(sizes) != len(ids):   # an empty layer: spread the size of the image evenly
            sizes = [sum(steps) // max(len(ids), 1)] * len(ids)
        layers.update(zip(ids, sizes))
    new = {layer: size for layer, size in layers.items() if layer not in seen}
    seen.update(layers)
    print(tag, len(layers), len(new), sum(new.values()))
EOF
)"
  human() { awk -v b="$1" 'BEGIN { split("B kB MB GB TB", u); i = 1
    while (b >= 1000 && i < 5) { b /= 1000; i++ }; printf(i == 1 ? "%d %s" : "%.1f %s", b, u[i]) }'; }
  clock() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }
  count=0 total=0
  while read -r tag layers new bytes; do count=$((count + 1)) total=$((total + bytes)); done <<< "$weights"
  echo "Push plan: $count tags, about $(human "$total") of layers (uncompressed; the registry gets them"
  echo "compressed, and skips layers it already has, so the push can be faster than this):"
  i=0
  while read -r tag layers new bytes; do
    i=$((i + 1))
    if [ "$new" = 0 ]; then extra="same layers as an earlier tag: manifests only"; else extra="$new new, about $(human "$bytes")"; fi
    printf '  %2d/%d %s: %d layers, %s\n' "$i" "$count" "$tag" "$layers" "$extra"
  done <<< "$weights"
  i=0 sent=0 start=$SECONDS
  while read -r target platforms tag; do
    i=$((i + 1))
    read -r _ layers new bytes <<< "$(sed -n "${i}p" <<< "$weights")"
    if [ "$dry_run" = true ]; then echo "would run: docker push $tag   ($platforms)"; continue; fi
    elapsed=$((SECONDS - start)) status=""
    if [ "$total" -gt 0 ]; then status="; overall $((sent * 100 / total)) % done"; fi
    if [ "$sent" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
      status="$status, elapsed $(clock "$elapsed"), about $(clock $(((total - sent) * elapsed / sent))) left"
    fi
    echo
    echo "[$i/$count] Pushing $tag ($platforms): $layers layers, $new new$status"
    docker push "$tag" < /dev/null   # the image index with every platform loaded here; docker shows each layer
    sent=$((sent + bytes)) elapsed=$((SECONDS - start))
    echo "[$i/$count] Pushed $tag; overall $([ "$total" -gt 0 ] && echo $((sent * 100 / total)) || echo 100) % done ($(human "$sent") of $(human "$total")), elapsed $(clock "$elapsed")"
  done <<< "$plan"
  if [ "$dry_run" != true ]; then
    elapsed=$((SECONDS - start))
    echo
    echo "Pushed $count tags in $(clock "$elapsed")$([ "$elapsed" -gt 0 ] && echo ", about $(human $((total / elapsed)))/s (uncompressed)")"
  fi
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
