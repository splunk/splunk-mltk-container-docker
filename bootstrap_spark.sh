#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

umask 002
cp -R -n /dltk/app /srv
cp -R -n /dltk/notebooks /srv
#if ! whoami &> /dev/null; then
  if [ -w /etc/passwd ]; then
    echo "dltk:x:$(id -u):0:dltk user:/dltk:/sbin/nologin" >> /etc/passwd
  fi
#fi
export HOME=/dltk

jupyter lab --port=8888 --ip=0.0.0.0 --no-browser & mlflow ui -p 6000 -h 0.0.0.0 & uvicorn app.main:app --host 0.0.0.0 --port 5000 --ssl-keyfile /dltk/.jupyter/dltk.key --ssl-certfile /dltk/.jupyter/dltk.pem
