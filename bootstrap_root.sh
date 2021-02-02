#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export FLASK_APP=/srv/app/index.py
export FLASK_DEBUG=1

jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root & tensorboard --logdir /srv/notebooks/logs/ & flask run -h 0.0.0.0
