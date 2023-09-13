#!/bin/bash
# build test env

if [ -d "./testing/venv" ]; then
    source ./testing/venv/bin/activate
else
    python3 -m venv ./testing/venv
    source ./testing/venv/bin/activate
    pip3 install playwright pytest pytest-playwright pyyaml
    playwright install
fi

if [ -z "$1" ]; then
  echo "No container tag specified."
  return 1
else
  container_tag="$1"
fi

files=$(find ./testing/test_mapping/ -type f | sort)
found_match=false

for file in $files; do
  echo $file
  echo $container_tag
  if grep -q "$container_tag" "$file"; then
    test_name=$(basename "$file")
    pytest ./testing/test_$test_name.py -q --containername $container_tag -v
    found_match=true
  fi
done

if [ "$found_match" = false ]; then
  echo "no tests found for this container"
fi