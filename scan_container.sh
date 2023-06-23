#!/bin/sh
# Compile python requirement versions for a specific image tag

# What does this script do?
# Scans a built image based on the tag, repo and version provided using a trivy container

# Usage and Requirements
# Simply run this script with the argument of an image tag in tag_mapping.csv
# e.g. ./scan_container.sh ubi-minimal-cpu
# optionally add a repo and version

if [ -z "$1" ]; then
  echo "No build tag specified. Pick a tag:"
  values=$(cut -d ',' -f 1 tag_mapping.csv)
  echo $values
  exit
else
  tag="$1"
fi

if [ -z "$2" ]; then
  repo="local/"
  echo "No repo name specified. Using default repo name: ${repo}"
else
  repo="$2"
fi

if [ -z "$3" ]; then
  version="latest"
  echo "No version specified. Using version: ${version}"
else
  version="$3"
fi

scan_time=$(date +%s)
container_name="$repo"mltk-container-$tag:$version
output_path=./scan_logs/${container_name////_}_${scan_time}.log

echo "Scanning $container_name"

docker run -q\
        -v /var/run/docker.sock:/var/run/docker.sock\
        aquasec/trivy:latest\
        image $container_name --scanners vuln\
        >> $output_path

echo "Scan finished, review output in $output_path"