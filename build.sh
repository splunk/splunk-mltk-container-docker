#!/bin/sh
echo "_____________________________________________________________________________________________________________"
echo ' __    __  __      ______ __  __       ______  ______  __   __  ______ ______  __  __   __  ______  ______    '
echo '/\ "-./  \/\ \    /\__  _/\ \/ /      /\  ___\/\  __ \/\ "-.\ \/\__  _/\  __ \/\ \/\ "-.\ \/\  ___\/\  == \   '
echo '\ \ \-./\ \ \ \___\/_/\ \\\ \  _"-.    \ \ \___\ \ \/\ \ \ \-.  \/_/\ \\\ \  __ \ \ \ \ \-.  \ \  __\\\ \  __<   '
echo ' \ \_\ \ \_\ \_____\ \ \_\\\ \_\ \_\    \ \_____\ \_____\ \_\\\"\_\ \ \_\\\ \_\ \_\ \_\ \_\\\"\_\ \_____\ \_\ \_\ '
echo '  \/_/  \/_/\/_____/  \/_/ \/_/\/_/     \/_____/\/_____/\/_/ \/_/  \/_/ \/_/\/_/\/_/\/_/ \/_/\/_____/\/_/ /_/ '
echo "_____________________________________________________________________________________________________________"
echo "Splunk> MLTK Container for TensorFlow 2.0, PyTorch and Jupyterlab."
tag="golden-image-gpu"
base="nvidia/cuda:10.2-cudnn7-runtime-ubuntu16.04"
dockerfile="Dockerfile"
repo="phdrieger/"
if [ -z "$1" ]; then
  echo "No build parameters set. Using default tag golden-image-gpu for building and running the container."
  echo "You can use ./build.sh [golden-image-gpu|tf-cpu|tf-gpu|pytorch|nlp] to build the container for different frameworks."
else
  tag="$1"
fi
case $tag in
	golden-image-cpu)
		base="ubuntu:20.04"
		dockerfile="Dockerfile.3.5.cpu"
		;;
	golden-image-gpu)
		base="nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04"
		dockerfile="Dockerfile.3.5"
		;;
	golden-image-gpu-3-4)
		base="nvidia/cuda:10.2-cudnn7-runtime-ubuntu16.04"
		;;
	spark)
		base="jupyter/pyspark-notebook:latest"
		dockerfile="Dockerfile.3.5.spark"
		;;
	spark-3-4)
		base="jupyter/pyspark-notebook:latest"
		dockerfile="Dockerfile.spark"
		;;
	rapids)
		base="rapidsai/rapidsai:cuda11.0-runtime-ubuntu16.04-py3.7"
		dockerfile="Dockerfile.3.5.rapids"
		;;
	rapids-3-4)
		base="rapidsai/rapidsai:0.17-cuda10.2-runtime-ubuntu16.04"
		dockerfile="Dockerfile.rapids"
		;;
	tf-cpu)
		base="tensorflow/tensorflow:2.0.0b1-py3"
		dockerfile="Dockerfile.root.3.0"
		;;
	tf-gpu)
		base="tensorflow/tensorflow:2.0.0b1-gpu-py3"
		dockerfile="Dockerfile.root.3.0"
		;;
	pytorch)
		base="pytorch/pytorch:latest"
		dockerfile="Dockerfile.root.3.0"
		;;
	nlp)
		base="tensorflow/tensorflow:2.0.0b1-gpu-py3"
		dockerfile="Dockerfile.root.3.0"
		;;
	*)
		echo "Invalid container image tag: $tag, expected [golden-image-gpu|tf-cpu|tf-gpu|pytorch|pytorch-nlp]"
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
docker build --rm -t "$repo"mltk-container-$tag:latest --build-arg BASE_IMAGE=$base --build-arg TAG=$tag -f $dockerfile .
if [ -z "$3" ]; then
  version=""
else
  version="$3"
  docker tag "$repo"mltk-container-$tag:latest "$repo"mltk-container-$tag:$version
fi
