#!/bin/bash
# Compile python requirement versions for a specific image tag

# What does this script do?
# Because of the large number of python packages that can be installed into DSDL containers, in some
# instances and with some base images it can sometimes take a long time (hours or days) to resolve all 
# of the python dependancies. This script creates a minimal image and uses it to pre-compute library 
# dependancies using pip-compile, which seems to be much faster and more robust.

# Usage and Requirements
# Simply run this script with the argument of an image tag in tag_mapping.csv
# e.g. ./compile_image_python_requirements.sh ubi-minimal-cpu
# A "Dockerfile.*.requirements" version must exist for the image tag's dockerfile. Please see 
# dockerfiles/Dockerfile.redhat.requirements for an example.
# For most of the images provided the requirements dockerfile will automatically be selected, if not
# Dockerfile.debian.requirements or Dockerfile.redhat.requirements are likely to be appropriate
# depending upon the specific OS base image used.

if [ -z "$1" ]; then
  echo "No build tag specified. Pick a tag:"
  values=$(cut -d ',' -f 1 tag_mapping.csv)
  echo $values
  exit
else
  tag="$1"
fi

echo "Compiling pip requirements for $tag"

line=$(grep "^${tag}," tag_mapping.csv)

echo $line

if [ "$line" != "" ]; then
    base_image=$(echo $line | cut -d',' -f2)
    
    if [[ -z "$dockerfile" ]]; then
      dockerfile=$(echo $line | cut -d',' -f3)
      dockerfile=$dockerfile.requirements
    fi

    base_requirements=$(echo $line | cut -d',' -f4)
    specific_requirements=$(echo $line | cut -d',' -f5)
    runtime=$(echo $line | cut -d',' -f6) 
    requirements_dockerfile=$(echo $line | cut -d',' -f7) 

    echo "Tag: $tag"
    echo "Base Image: $base_image"
    echo "Dockerfile: $dockerfile"
    echo "Base Requirements File: $base_requirements"
    echo "Specific Requirements File: $specific_requirements"
    echo "Runtime Options: $runtime"
    echo "Requirements Dockerfile: $requirements_dockerfile"

    base_requirements="${base_requirements%.*}"
    specific_requirements="${specific_requirements%.*}"

    compiled_requirements_id=compiled_${base_requirements}_${specific_requirements}_$tag
    compiled_requirements_filename=./requirements_files/$compiled_requirements_id.in

    rm -f $compiled_requirements_filename

    cat ./requirements_files/$base_requirements.txt >> $compiled_requirements_filename
    cat ./requirements_files/$specific_requirements.txt >> $compiled_requirements_filename
    
    image_name="$repo"mltk-container-$tag-requirements
    container_name=temp-requirements-compile-$tag

    echo "Building container to compute compiled requirements"
    docker build --rm -t $image_name \
        --build-arg BASE_IMAGE=$base_image \
        --build-arg TAG=$tag \
        --build-arg COMPILED_REQUIREMENTS_FILE=$compiled_requirements_id.in \
        -f ./dockerfiles/$requirements_dockerfile\
        .

    echo "Running container to copy compiled requirements"
    docker run --name $container_name $image_name
    docker cp $container_name:/temp/$compiled_requirements_id.txt ./requirements_files/$compiled_requirements_id.txt

    docker stop $container_name
    docker rm $container_name
    #docker image rm $image_name
fi

