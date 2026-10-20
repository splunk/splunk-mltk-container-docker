#!/bin/sh
# Compatibility entrypoint: /dltk/bootstrap_fast.sh has been the entrypoint path of
# every DSDL image since 5.0. The start-up logic lives in /dltk/entrypoint.py.
exec /opt/dltk/venv/bin/python -I /dltk/entrypoint.py "$@"
