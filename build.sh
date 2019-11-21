#!/bin/sh
echo "_____________________________________________________________________________________________________________"
echo ' __    __  __      ______ __  __       ______  ______  __   __  ______ ______  __  __   __  ______  ______    '
echo '/\ "-./  \/\ \    /\__  _/\ \/ /      /\  ___\/\  __ \/\ "-.\ \/\__  _/\  __ \/\ \/\ "-.\ \/\  ___\/\  == \   '
echo '\ \ \-./\ \ \ \___\/_/\ \\\ \  _"-.    \ \ \___\ \ \/\ \ \ \-.  \/_/\ \\\ \  __ \ \ \ \ \-.  \ \  __\\\ \  __<   '
echo ' \ \_\ \ \_\ \_____\ \ \_\\\ \_\ \_\    \ \_____\ \_____\ \_\\\"\_\ \ \_\\\ \_\ \_\ \_\ \_\\\"\_\ \_____\ \_\ \_\ '
echo '  \/_/  \/_/\/_____/  \/_/ \/_/\/_/     \/_____/\/_____/\/_/ \/_/  \/_/ \/_/\/_/\/_/\/_/ \/_/\/_____/\/_/ /_/ '
echo "_____________________________________________________________________________________________________________"
echo "Splunk> MLTK Container for TensorFlow 2.0, PyTorch and Jupyterlab."
tag="tf-cpu"
base="tensorflow/tensorflow:latest-py3"
repo=""
if [ -z "$1" ]; then
  echo "No build parameters set. Using default tag tf-cpu for building and running the container."
  echo "You can use ./build.sh [tf-cpu|tf-gpu|pytorch|nlp] to build the container for different frameworks."
else
  tag="$1"
fi
case $tag in
	tf-cpu)
		base="tensorflow/tensorflow:2.0.0b1-py3"
		;;
	tf-gpu)
		base="tensorflow/tensorflow:2.0.0b1-gpu-py3"
		;;
	pytorch)
		base="pytorch/pytorch:latest"
		;;
	nlp)
		base="tensorflow/tensorflow:2.0.0b1-gpu-py3"
		;;
	rapidsai)
		base="continuumio/miniconda"
		;;
	*)
		echo "Invalid container image tag: $tag, expected [tf-cpu|tf-gpu|pytorch|pytorch-nlp]"
    	break
		;;
esac
if [ -z "$2" ]; then
  echo "No target repo set, using default prefix phdrieger/"
  repo="phdrieger/"
else
  repo="$2"
fi
echo "Using tag [$tag] for building container based on [$base]"
echo "Stop, remove and build container..."
docker stop "$repo"mltk-container-$tag
docker rm "$repo"mltk-container-$tag
docker rmi "$repo"mltk-container-$tag
docker build --rm -t "$repo"mltk-container-$tag:latest --build-arg BASE_IMAGE=$base --build-arg TAG=$tag -f Dockerfile .
