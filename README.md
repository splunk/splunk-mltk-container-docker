# Splunk App for Data Science and Deep Learning

Splunk App for Data Science and Deep Learning (DSDL) 5.0.0
formerly known as Deep Learning Toolkit for Splunk (DLTK) versions 2.3.0 - 3.9.0 published on [splunkbase](https://splunkbase.splunk.com/app/4607/)
and DLTK version 4.x open sourced on [GitHub](https://github.com/splunk/deep-learning-toolkit)

Copyright (C) 2005-2022 Splunk Inc. All rights reserved.  
Author: [Philipp Drieger]()

## Container Images

This repository contains the container endpoint (`./app`), jupyter notebook configuration (`./config`) and examples (`./notebooks`), build scripts and the main Dockerfiles to create the [existing pre-built container images](https://hub.docker.com/u/phdrieger) for TensorFlow, PyTorch, NLP libraries and many other data science libraries for CPU and GPU.

### Rebuild 
You can rebuild your own containers with the `build.sh` script.

Examples:
- Build Golden Image CPU image for your own docker repo
`./build.sh golden-image-cpu your_local_docker_repo/ 5.0.0`

- Build Golden Image GPU image for your own docker repo
`./build.sh golden-image-gpu your_local_docker_repo/ 5.0.0`

If you decide to modify to `your_local_docker_repo/` you need to update your `images.conf` in the DSDL app: go to your `$SPLUNK_HOME/etc/apps/mltk-container/local/images.conf` and add your own image stanzas. Have a look at `$SPLUNK_HOME/etc/apps/mltk-container/default/images.conf` to see how the stanzas are defined.

### Build your own custom container images
Feel free to extend the build script and Dockerfile to create your own custom MLTK Container images.
To make your own images available in the DSDL app, please add a local config file to the app: go to your `$SPLUNK_HOME/etc/apps/mltk-container/local/images.conf` and add for example your new stanza:

[myimage]\
title = My custom image\
image = mltk-container-myimage\
repo = your_local_docker_repo/\
runtime = none,nvidia\

### Certificates
For development purposes the container images contain a self-signed certificate for HTTPS. You can replace the `dltk.key` and `dltk.pem` files in the `config` folder and build the container. This is one possibility to use your own certificates. There are also other options to configure your container environment with your own certificates.

### Run and test your container locally
You can run your container locally, e.g. with `docker run -it --rm --name mltk-container-golden-image-cpu -p 5000:5000 -p 8888:8888 -p 6006:6006 -v mltk-container-data:/srv phdrieger/mltk-container-golden-image-cpu:5.0.0`

## Further documentation and usage

Please find further information and documentation on splunkbase: [Download and install the Splunk App for Data Science and Deep Learning](https://splunkbase.splunk.com/app/4607/)
