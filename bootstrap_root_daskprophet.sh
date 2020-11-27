#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export FLASK_APP=/srv/app/index.py
export FLASK_DEBUG=0

jupyter lab --port=8888 --ip=0.0.0.0 --no-browser & flask run -h 0.0.0.0
