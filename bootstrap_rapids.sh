#!/bin/bash
source activate rapids
#ipython kernel install --user --name=rapids

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

jupyter lab --no-browser & python -m app.main
