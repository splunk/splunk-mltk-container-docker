#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export FLASK_APP=/srv/app/index.py
export FLASK_DEBUG=0

jupyter lab --no-browser & tensorboard --bind_all --logdir /srv/notebooks/logs/ & flask run 
