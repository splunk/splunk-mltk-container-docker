#!/usr/bin/env bash
# Pull every published DSDL image reference listed in images.toml, e.g. to mirror
# them into a private registry for an air-gapped installation.
#
# Usage: tools/pull.sh [repo] [version]        default: splunk/ and the images.toml version
set -Eeuo pipefail
cd "$(dirname "$0")/.."
repo="${1:-splunk/}"
version="${2:-$(sed -n 's/^version = "\(.*\)"/\1/p' images.toml)}"
if python3 -c 'import sys; sys.version_info >= (3, 11) or __import__("tomli")' 2>/dev/null; then
  refs="$(python3 tools/render.py --references --repo "$repo" --version "$version")"
else
  refs="$(docker run --rm -v "$PWD:/repo" -w /repo registry.access.redhat.com/ubi9/ubi \
    bash -c 'dnf -yq install python3.12 >/dev/null && python3.12 tools/render.py --references --repo "$0" --version "$1"' "$repo" "$version")"
fi
for ref in $refs; do
  echo "Pulling $ref"
  docker pull "$ref"
done
