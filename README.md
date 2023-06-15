# Splunk App for Data Science and Deep Learning

Splunk App for Data Science and Deep Learning (DSDL) 5.1.0
formerly known as Deep Learning Toolkit for Splunk (DLTK) versions 2.3.0 - 3.9.0 published on [splunkbase](https://splunkbase.splunk.com/app/4607/)
and DLTK version 4.x open sourced on [GitHub](https://github.com/splunk/deep-learning-toolkit)

Copyright (C) 2005-2023 Splunk Inc. All rights reserved.  
Author: [Philipp Drieger]()
Contributors: [Josh Cowling](https://www.linkedin.com/in/josh-cowling/)

## Container Images

This repository contains the container endpoint (`./app`), jupyter notebook configuration (`./config`) and examples (`./notebooks`), build scripts and the main Dockerfiles to create the [existing pre-built container images](https://hub.docker.com/u/phdrieger) for TensorFlow, PyTorch, NLP libraries and many other data science libraries for CPU and GPU.

### Rebuild 
You can rebuild your own containers with the `build.sh` script.

The `build.sh` script is invoked with at least one and up to three arguments:
`./build.sh <build_configuration_tag> <repo_name> <version>`

`<build_configuration_tag>` is used to specify the particular set of base docker image, dockerfile, base python requirements, specific python requirements and runtime setting.
These combinations can be found and added to in `tag_mapping.csv`.

`<repo_name>` allows you to specify a repo prefix which will be needed if you intend to upload images to dockerhub.

`<version>` allows you to specify a new version number for the completed image.

To build the default golden cpu image locally, simply run:
`./build.sh golden-cpu`

or specify additional arguments:
`./build.sh golden-cpu local_build/ 1.0.0`

In this latest version the following combinations are available, but customizations can easily be made and added to `tag_mapping.csv` file for custom builds:

| `build configuration tag` | Description |
| --- | --- |
| minimal-cpu | Debian bullseye based with a minimal data-science environment (numpy,scipi,pandas,scikit-learn,matplotlib,etc). Notably this does not include tensorflow or pytorch which significantly bloat image size. |
| minimal-gpu | Debian bullseye based with a minimal data-science environment (numpy,scipi,pandas,scikit-learn,matplotlib,etc). Notably this does not include tensorflow or pytorch which significantly bloat image size. Does include jupyter nvidia dashboards for GPU resource monitoring. |
| golden-cpu | Debian bullseye based with a wide range of data science libraries and tools. (all of the above including tensorflow, pytorch, umap-learn, datashader, dask, spacy, networkx and many more see `requirements_files/specific_golden_cpu.txt` for more details). Excludes tensorflow and torch GPU functionality where possible | 
| golden-gpu | The same as golden-cpu but with tensorflow and pytorch GPU libraries |
| ubi-functional-cpu | Redhat UBI9 based image. Contains only the specific libraries needed to have a functional conntection between the DSDL app and an external container. Most suitable for building custom enterprise-ready images on top of. |
| ubi-minimal-cpu | Redhat UBI9 based image with a basic data science environment. |
| ubi-golden-cpu | Redhat UBI9 based image with a wide range of data science libraries and tools. |
| golden-cpu-milvus | Variation on the Debian golden CPU image which contains the pymilvus library for use with push_to_milvus and query_milvus notebooks and can be used to integrate Splunk with the open-source vector database Milvus. |
| golden-cpu-transformers | Variation on the Debian golden CPU image which supports the use of certain transformer models |
| golden-gpu-transformers | Variation on the Debian golden CPU image which supports the use of certain transformer models, including GPU suppport |

When building images the build script creates an images.conf which will be placed in the `<splunk>/etc/apps/mltk-container/local` directory to make the image available for use in the DSDL app.

### Build your own custom container images
Feel free to extend the build script and Dockerfiles to create your own custom MLTK Container images.

To do this add an entry to the `tag_mapping.csv` file which references a base image, dockerfile, requirements files and a runtime context.
Example requirements files and dockerfiles can be found in `requirements_files/` and `dockerfiles/` directories.

### Certificates
For development purposes the container images create self-signed certificates for HTTPS. You can replace the `dltk.key` and `dltk.pem` filed by editing the Dockerfile to build a container with your own certificates. This is one possibility to use your own certificates. There are also other options to configure your container environment with your own certificates.

### Run and test your container locally
You can run your container locally, e.g. with `docker run -it --rm --name mltk-container-golden-image-cpu -p 5000:5000 -p 8888:8888 -p 6006:6006 -v mltk-container-data:/srv phdrieger/mltk-container-golden-image-cpu:5.0.0`

## Further documentation and usage

Please find further information and documentation on splunkbase: [Download and install the Splunk App for Data Science and Deep Learning](https://splunkbase.splunk.com/app/4607/)
