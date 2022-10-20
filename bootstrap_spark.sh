#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

umask 002
cp -R -n /dltk/app /srv
cp -R -n /dltk/notebooks /srv
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
  echo "Starting in mode = PROD"
  uvicorn app.main:app --host 0.0.0.0 --port 5000 $uvicorn_https_param
else
  echo "Starting in mode = DEV"
  jupyter lab --no-browser --ip=0.0.0.0 --port=8888 & mlflow ui -p 6000 -h 0.0.0.0 & uvicorn app.main:app --host 0.0.0.0 --port 5000 $uvicorn_https_param
fi

