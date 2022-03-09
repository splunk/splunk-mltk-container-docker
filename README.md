# Deep Learning Toolkit for Splunk (2.x and 3.x)

Deep Learning Toolkit for Splunk (version 2.3.0 - 3.8.0)
Copyright (C) 2005-2021 Splunk Inc. All rights reserved.  
Author: [Philipp Drieger]()

For the latest development please check out [DLTK version 4.x available on GitHub](https://github.com/splunk/deep-learning-toolkit).

## MLTK Container

This repository contains the container endpoint (`./app`), jupyter notebook configuration (`./config`) and examples (`./notebooks`), build scripts and the main Dockerfile to create the [existing pre-built container images](https://hub.docker.com/u/phdrieger) for TensorFlow 2.0 CPU and GPU, PyTorch CPU and GPU, NLP libraries.

### Rebuild 
You can rebuild your own containers with the `build.sh` script.

Examples:
- Build Golden Image CPU image for your own docker repo
`./build.sh golden-image-cpu your_local_docker_repo/ 3.8.0`

- Build Golden Image GPU image for your own docker repo
`./build.sh golden-image-gpu your_local_docker_repo/ 3.8.0`

If you decide to modify to `your_local_docker_repo/` you need to update your `images.conf` in the Deep Learning Toolkit app: go to your `$SPLUNK_HOME/etc/apps/mltk-container/local/images.conf` and add your own image stanzas. Have a look at `$SPLUNK_HOME/etc/apps/mltk-container/default/images.conf` to see how the stanzas are defined.

### Build your own custom container images
Feel free to extend the build script and Dockerfile to create your own custom MLTK Container images.
To make your own images available in the Deep Learning Toolkit app, please add a local config file to the app: go to your `$SPLUNK_HOME/etc/apps/mltk-container/local/images.conf` and add for example your new stanza:

[myimage]\
title = My custom image\
image = mltk-container-myimage\
repo = your_local_docker_repo/\
runtime = none,nvidia\

### Certificates
For development purposes the container images contain a self-signed certificate for HTTPS. You can replace the `dltk.key` and `dltk.pem` files in the `config` folder and build the container. This is one possibility to use your own certificates. There are also other options to configure your container environment with your own certificates.

## Further documentation and usage

Please find further information and documentation contained in the Deep Learning Toolkit app on splunkbase: [Download and install the Deep Learning Toolkit](https://splunkbase.splunk.com/app/4607/)
