#!/bin/sh
# build test env

if [ -d "./testing/venv" ]; then
    source ./testing/venv/bin/activate
else
    python3 -m venv ./testing/venv
    source ./testing/venv/bin/activate
    pip3 install playwright
fi

playwright install