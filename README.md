# Deep Learning Toolkit for Splunk 

Deep Learning Toolkit for Splunk (version 2.3.0)
Copyright (C) 2005-2019 Splunk Inc. All rights reserved.  
Author: [Philipp Drieger]()

## MLTK Container

This repository contains the container endpoint (`./app`), jupyter notebook configuration (`./config`) and examples (`./notebooks`), build scripts and the main Dockerfile to create the [existing pre-built container images](https://hub.docker.com/u/phdrieger) for TensorFlow 2.0 CPU and GPU, PyTorch CPU and GPU, NLP libraries.

### Rebuild 
You can rebuild your own containers with the `build.sh` script. Examples:

- Build TensorFlow CPU image for your own docker repo
`./build.sh tf-cpu your_local_docker_repo/`

- Build TensorFlow GPU image for your own docker repo
`./build.sh tf-gpu your_local_docker_repo/`

- Build PyTorch image for your own docker repo
`./build.sh pytorch your_local_docker_repo/`

- Build NLP image for your own docker repo
`./build.sh nlp your_local_docker_repo/`

If you decide to modify to `your_local_docker_repo/` you need to update your `images.conf` in the Deep Learning Toolkit app: go to your `$SPLUNK_HOME/etc/apps/mltk-container/local/images.conf` and add your own image stanzas. Have a look at `$SPLUNK_HOME/etc/apps/mltk-container/default/images.conf` to see how the stanzas are defined.

### Build your own custom container images
Feel free to extend the build script and Dockerfile to create your own custom MLTK Container images.
To make your own images available in the Deep Learning Toolkit app, please add a local config file to the app: go to your `$SPLUNK_HOME/etc/apps/mltk-container/local/images.conf` and add for example your new stanza:

[myimage]
title = My custom image
image = mltk-container-myimage
repo = your_local_docker_repo/
runtime = none,nvidia

## Further documentation and usage

Please find further information and documentation contained in the Deep Learning Toolkit app in the overview section. [Download and install the Deep Learning Toolkit](https://splunkbase.splunk.com/app/4607/)