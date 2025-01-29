# Splunk App for Data Science and Deep Learning

Splunk App for Data Science and Deep Learning (DSDL) 5.2.0
formerly known as Deep Learning Toolkit for Splunk (DLTK) versions 2.3.0 - 3.9.0 and DSDL 5.0.0 - 5.1.2 published on [splunkbase](https://splunkbase.splunk.com/app/4607/)
and DLTK version 4.x open sourced on [GitHub](https://github.com/splunk/deep-learning-toolkit)

Copyright (C) 2005-2024 Splunk Inc. All rights reserved.  
Author: [Philipp Drieger]()
Contributors: [Josh Cowling](https://www.linkedin.com/in/josh-cowling/), [Huaibo Zhao](), [Tatsu Murata]() 

# About:
This repository contains the collection of resources, scripts, and testing frameworks that are used build and deploy the default container images used by the DSDL app. It can be used to modify, secure, update and change these containers and their build process to meet the needs of your enterprise environment.

## Resources
This repository contains the container endpoint (`./app`), jupyter notebook configuration (`./config`) and examples (`./notebooks`), build scripts and the main Dockerfiles to create the [existing pre-built container images](https://hub.docker.com/u/phdrieger) for TensorFlow, PyTorch, NLP libraries and many other data science libraries for CPU and GPU.

### Building containers 
You can  your own containers with the `build.sh` script.

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

In this latest version of this repository the following tags are available, but customizations can easily be made and added to `tag_mapping.csv` file for custom builds:

| `build configuration tag` | Description |
| --- | --- |
| minimal-cpu | Debian bullseye based with a minimal data-science environment (numpy,scipi,pandas,scikit-learn,matplotlib,etc). Notably this does not include tensorflow or pytorch which significantly bloat image size. |
| minimal-gpu | Debian bullseye based with a minimal data-science environment (numpy,scipi,pandas,scikit-learn,matplotlib,etc). Notably this does not include tensorflow or pytorch which significantly bloat image size. Does include jupyter nvidia dashboards for GPU resource monitoring. |
| golden-cpu | Debian bullseye based with a wide range of data science libraries and tools. (all of the above including tensorflow, pytorch, umap-learn, datashader, dask, spacy, networkx and many more see `requirements_files/specific_golden_cpu.txt` for more details). Excludes tensorflow and torch GPU functionality where possible | 
| golden-gpu | The same as golden-cpu but with tensorflow and pytorch GPU libraries |
| ubi-functional-cpu | Redhat UBI9 based image. Contains only the specific libraries needed to have a functional conntection between the DSDL app and an external container. Most suitable for building custom enterprise-ready images on top of. |
| ubi-minimal-cpu | Redhat UBI9 based image with a basic data science environment. |
| ubi-golden-cpu | Redhat UBI9 based image with a wide range of data science libraries and tools. Spacy excluded due to build issues on redhat. |
| golden-gpu-transformers | Variation on the Debian golden CPU image which supports the use of certain transformer models, including GPU suppport |
| golden-gpu-rapids | Variation on the Debian golden CPU image which supports the use of rapids on a GPU enabled image |

When building images the build script creates an images.conf which will be placed in the `<splunk>/etc/apps/mltk-container/local` directory to make the image available for use in the DSDL app.

### Build your own custom container images
You may extend the resources available to create your own images.

To do this add an entry to the `tag_mapping.csv` file which references a `base image`, `dockerfile`, `requirements files` and a `runtime` context.

`tag_mapping.csv` columns:
| Column | Description | Notes |
| --- | --- | --- |
| Tag | A short name given to a combination of resources that make up an image | |
| base_image | The base image to build a DSDL container from | This may be your operating system of choice, or one that specficially supports some functionality, such as GPU libraries. This is usually gathered from a public images repository such as Dockerhub, or a private image repository. |
| dockerfile | The dockerfile to use to build this container. | Dockerfiles must be placed in the `./dockerfiles/` directory. |
| base_requirements / specific_requirements | These columns specify the names of requirements files to use from the `./requirements_files/` directory | For consistency  python requirements files are spit into two. For most images `base_requirements` can be set to use the `base_functional.txt` requirements file, which contains all of the basic libraries needed for DSDL to function. `specifc_requirements` may then be set to install libraries appropriate for your particualr use-case or environment. For the majority of examples in the DSDL app to function you must have the libraries listed in `specific_golden_cpu.txt` installed. Note: the build scipt will only use these files if a compiled requirements file has not been created for that image. If you make changes to an existing requirements file please ensure you delete any compiled requirements files (./requirements_files/compiled_*) before beginning a new build.
| runtime | This may be set to only two values: `none` or `nvidia` | `nvidia` only required if you are intending to use GPU functionality |
| requirements_dockerfile | This is only used if you are pre-compiling python requirements. Set this to a dockerfile with a minimal environment which will work with the proposed base image. Debian and Redhat variants are provided, see `./dockerfiles/Dockerfile.*.(redhat\|debian).requirements` for examples. Others may be added as needed. | |

Example requirements files and dockerfiles can be found in `requirements_files/` and `dockerfiles/` directories.

### Supporting Scripts
There are a number of scripts in this repo which can help in various tasks when working with DSDL containers and collections of containers. Information about these is provided in summary in this table and in more detail below.

| Script Name | Description | Example | Notes |
| --- | --- | --- | --- |
| `build.sh` | Build a container using a configuration tag found in `tag_mapping.csv` | `./build.sh minimal-cpu splunk/ 5.1.1` | |
| `bulk_build.sh` | Build all containers in a tag list | `./bulk_build.sh tag_mapping.csv splunk/ 5.1.1` | |
| `compile_image_python_requirements.sh` | Use a base image and simplified dockerfile to pre-compute the python dependancy versions for all libraries listed in the tag's referenced requirements files | `./compile_image_python_requirements.sh minimal-cpu` | If the Dockerfile for the tag is not specified, the script looks for the tags Dockerfile plus the `.requirements` extension. If this does not exist, please create a requirements dockerfile or specifiy and appropriate requirements dockerfile. An example can be found in /dockerfiles/Dockerfile.debian.requirements |
| `bulk_compile.sh` | Attempt to pre-compile python dependancy versions for all containers in a tag list | `./bulk_build.sh tag_mapping.csv` | Makes assumptions about dockerfile names as described above. |
| `scan_container.sh` | Scan a built container for vulnerabilities and produce a report with Trivy | `./scan_container.sh minimal-cpu splunk/ 5.1.1` | Downloads the Trivy container to run the scan. |
| `test_container.sh` | Run a set of simulated tests using Playwright on a built container. | `./test_container.sh minimal-cpu splunk/ 5.1.1` | Requires the setup of a python virtual environment that can run Playwright. Specific python versions and dependancies may be required at the system level. |

### Compile requirements for an image
In some cases, dependancy resolution can take a long time for an image with many python libraries. In this case you may find it faster to pre-compile python requirements. `compile_image_python_requirements.sh` and `bulk_compile.sh` scripts are provided for you to do this. Most pre-existing images will have compiled requirements shipped in this repo. If you need to rebuild an existing container with new libraries, please delete the associated compiled requirements file.

### Configuraing DSDL with images.conf
After running the `./build.sh` or `./bulk_build.sh` scripts, `<tag_name>.conf` files will be created in the `./images_conf_files/` directory. If deploying a single image you may move the contents of `<tag_name>.conf` into `/mltk-container/local/images.conf` in your Splunk installation, or you may copy the contents of `./images_conf_files/images.conf` into the app to configure all of the built images. 

Note: for an image to be deployed it must be made available to the docker or k8s instance by publishing to a public repository (default is dockerhub) or by adding the images to an appropriate private repository.

### Certificates
For development purposes the container images create self-signed certificates for HTTPS. You can replace the `dltk.key` and `dltk.pem` filed by editing the Dockerfile to build a container with your own certificates. This is one possibility to use your own certificates. There are also other options to configure your container environment with your own certificates.

### Run and test your container locally
You can run your container locally, e.g. with `docker run -it --rm --name mltk-container-golden-image-cpu -p 5000:5000 -p 8888:8888 -p 6006:6006 -v mltk-container-data:/srv phdrieger/mltk-container-golden-image-cpu:5.0.0`

## Further documentation and usage
Please find further information and documentation on splunkbase: [Download and install the Splunk App for Data Science and Deep Learning](https://splunkbase.splunk.com/app/4607/)
