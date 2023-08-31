#!/bin/sh
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

pytest ./testing/ -q --containername $container_tag