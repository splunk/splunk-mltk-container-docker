#!/usr/bin/env bash
# Resolve, check or upgrade the hashed uv locks of the dependency profiles.
#
# Usage:
#   tools/lock.sh                 # resolve all profiles (keeps existing pins where possible)
#   tools/lock.sh cpu llm         # resolve selected profiles
#   tools/lock.sh --check         # fail if any uv.lock, or a cpu-constraint block, is out of date
#   tools/lock.sh --upgrade cpu   # move every package to the newest allowed version
#   tools/lock.sh --upgrade-package=pip cpu
#                                 # move one package (e.g. for a CVE fix) and keep all other pins;
#                                 # repeat the option to move several
#
# When cpu is (re)locked, the cpu-lock constraint blocks of the gpu-cu12, rapids-cu12 and
# llm profiles are regenerated (tools/sync-cpu-constraints.py) and those profiles are
# re-locked as well, so shared packages keep the cpu versions.
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
    --upgrade-package=*) mode+=(--upgrade-package "${arg#*=}") ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) profiles+=("$arg") ;;
  esac
done
if [ "${#profiles[@]}" -eq 0 ]; then
  profiles=()
  for dir in deps/*/; do profiles+=("$(basename "$dir")"); done
fi

derived=(gpu-cu12 rapids-cu12 llm)
# cpu first; when cpu is locked, the derived profiles follow
ordered=()
for profile in "${profiles[@]}"; do [ "$profile" = cpu ] && ordered+=(cpu); done
for profile in "${profiles[@]}"; do [ "$profile" != cpu ] && ordered+=("$profile"); done
case " ${ordered[*]} " in
  *" cpu "*) for profile in "${derived[@]}"; do
               case " ${ordered[*]} " in *" $profile "*) ;; *) ordered+=("$profile") ;; esac
             done ;;
esac

builder_tag="dsdl-builder:lock"
docker buildx build --load --quiet --target builder -t "$builder_tag" -f docker/Dockerfile . >/dev/null
# run as the calling user so files written into the repository keep their owner on Linux hosts
as_user=(--user "$(id -u):$(id -g)" -e HOME=/tmp -e UV_CACHE_DIR=/uv-cache -v dsdl-uv-cache:/uv-cache)

failed=0
for profile in "${ordered[@]}"; do
  if [ ! -f "deps/$profile/pyproject.toml" ]; then
    echo "Unknown profile: $profile" >&2
    exit 2
  fi
  for derived_profile in "${derived[@]}"; do
    if [ "$profile" = "$derived_profile" ]; then
      sync_args=("$profile")
      if [ "${mode[*]+${mode[*]}}" = "--check" ]; then sync_args=(--check "$profile"); fi
      docker run --rm "${as_user[@]}" -v "$PWD:/repo:ro" -v "$PWD/deps/$profile:/repo/deps/$profile" \
        -w /repo "$builder_tag" python3.12 tools/sync-cpu-constraints.py "${sync_args[@]}" || failed=1
    fi
  done
  echo "uv lock ${mode[*]+${mode[*]}} -> deps/$profile/uv.lock"
  # the repo is mounted read-only (path sources such as ../../package-dsdlsupport);
  # only the profile directory is writable so uv can write uv.lock
  docker run --rm "${as_user[@]}" \
    -v "$PWD:/repo:ro" -v "$PWD/deps/$profile:/repo/deps/$profile" -w "/repo/deps/$profile" \
    "$builder_tag" uv lock ${mode[@]+"${mode[@]}"} || failed=1
done
exit "$failed"
