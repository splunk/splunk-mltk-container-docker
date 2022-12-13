#!/bin/sh
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock --net jupyterhub --name jupyterhub -p 8000:8000 phdrieger/mltk-container-jupyterhub:latest