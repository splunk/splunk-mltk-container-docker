#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

umask 002
/dltk/bootstrap_backup.sh
cp -R /dltk/app /srv
cp -R /dltk/notebooks /srv
if [ -w /etc/passwd ]; then
  echo "dltk:x:$(id -u):0:dltk user:/dltk:/sbin/nologin" >> /etc/passwd
fi
export HOME=/dltk

uvicorn_https_param="--ssl-keyfile /dltk/.jupyter/dltk.key --ssl-certfile /dltk/.jupyter/dltk.pem" 
if [ "$ENABLE_HTTPS" = "false" ]; then
  uvicorn_https_param=""
else
  echo "ENABLE_HTTPS=true"
fi

if [ "$MODE_DEV_PROD" = "PROD" ]; then
  if [ -n "$api_workers" ]; then
    echo "Starting in mode = PROD with $api_workers api workers"
    uvicorn app.main:app --host 0.0.0.0 --workers $api_workers --port 5000 $uvicorn_https_param
  else
    echo "Starting in mode = PROD with 1 api workers"
    uvicorn app.main:app --host 0.0.0.0 --workers 1 --port 5000 $uvicorn_https_param
  fi
else
  if [ -n "$api_workers" ]; then
    echo "Starting in mode = DEV with $api_workers api workers"
    jupyter lab --no-browser --ip=0.0.0.0 --port=8888 & tensorboard --bind_all --logdir /srv/notebooks/logs/ & mlflow ui -p 6000 -h 0.0.0.0 & uvicorn app.main:app --host 0.0.0.0 --workers $api_workers --port 5000 --reload-dir app $uvicorn_https_param
  else
    echo "Starting in mode = DEV with default 1 api workers"
    jupyter lab --no-browser --ip=0.0.0.0 --port=8888 & tensorboard --bind_all --logdir /srv/notebooks/logs/ & mlflow ui -p 6000 -h 0.0.0.0 & uvicorn app.main:app --host 0.0.0.0 --workers 1 --port 5000 --reload-dir app $uvicorn_https_param
  fi
fi
