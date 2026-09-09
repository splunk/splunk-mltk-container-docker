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

    # Pick the build target platform (mirrors build.sh). The requirements
    # compile container resolves wheels for this platform — so the resulting
    # lockfile is platform-correct for the eventual runtime image. Tags ending
    # in "-arm" → linux/arm64; everything else → linux/amd64 (with qemu
    # emulation on Apple Silicon dev machines).
    case "$tag" in
      *-arm) build_platform="linux/arm64" ;;
      *)     build_platform="linux/amd64" ;;
    esac

    echo "Tag: $tag"
    echo "Base Image: $base_image"
    echo "Dockerfile: $dockerfile"
    echo "Base Requirements File: $base_requirements"
    echo "Specific Requirements File: $specific_requirements"
    echo "Runtime Options: $runtime"
    echo "Requirements Dockerfile: $requirements_dockerfile"
    echo "Build Platform: $build_platform"

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
    if ! docker build --rm --platform=$build_platform -t $image_name \
        --build-arg BASE_IMAGE=$base_image \
        --build-arg TAG=$tag \
        --build-arg COMPILED_REQUIREMENTS_FILE=$compiled_requirements_id.in \
        -f ./dockerfiles/$requirements_dockerfile\
        .; then
      echo "Failed to build the requirements compiler image for $tag." >&2
      exit 1
    fi

    echo "Running container to copy compiled requirements"
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    # The lock is generated during the image build. Create (but do not start)
    # a container so inherited entrypoints such as Jupyter cannot block the
    # copy step.
    if ! docker create --platform=$build_platform --name $container_name $image_name >/dev/null; then
      echo "Failed to create the requirements copy container for $tag." >&2
      docker rm -f "$container_name" >/dev/null 2>&1 || true
      exit 1
    fi
    if ! docker cp "$container_name:/temp/$compiled_requirements_id.txt" "./requirements_files/$compiled_requirements_id.txt"; then
      echo "Failed to copy the compiled requirements for $tag." >&2
      docker rm -f "$container_name" >/dev/null 2>&1 || true
      exit 1
    fi

    docker rm $container_name
    #docker image rm $image_name
fi
