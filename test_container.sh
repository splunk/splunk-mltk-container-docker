#!/bin/bash
# Run the Playwright UI tests (testing/) for one images.conf stanza against a live
# Splunk instance configured in testing/config.yaml. The container is selected in
# the DSDL app by its stanza title from images.toml.
#
# Usage: ./test_container.sh <stanza> [version]      e.g. ./test_container.sh golden-cpu 5.3.0
cd "$(dirname "$0")"

if [ -d "./testing/venv" ]; then
    source ./testing/venv/bin/activate
else
    python3 -m venv ./testing/venv
    source ./testing/venv/bin/activate
    pip3 install playwright pytest pytest-playwright pyyaml tomli
    playwright install
fi

if [ -z "$1" ]; then
  echo "No stanza specified. Stanzas: $(python3 tools/render.py --stanzas 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
stanza="$1"
version_args=()
if [ -n "$2" ]; then version_args=(--version "$2"); fi
title="$(python3 tools/render.py --title "$stanza" "${version_args[@]}")" || exit 1
echo "Stanza: $stanza  Title: $title"

found_match=false
for file in $(find ./testing/test_mapping/ -type f | sort); do
  if grep -Fxq "$stanza" "$file"; then
    test_name=$(basename "$file")
    pytest ./testing/test_$test_name.py -q --containername "$title" -v
    found_match=true
  fi
done

if [ "$found_match" = false ]; then
  echo "no tests found for this container"
fi
