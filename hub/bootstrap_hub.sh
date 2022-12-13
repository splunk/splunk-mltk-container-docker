#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

#jupyterhub --ip 10.0.1.2 --port 443 --ssl-key my_ssl.key --ssl-cert my_ssl.cert
jupyterhub --ip 0.0.0.0 --port 8000 -f /etc/jupyterhub/jupyterhub_config.py