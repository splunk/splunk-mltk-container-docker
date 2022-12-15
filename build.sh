#!/bin/sh
echo "_____________________________________________________________________________________________________________"
echo ' __    __  __      ______ __  __       ______  ______  __   __  ______ ______  __  __   __  ______  ______    '
echo '/\ "-./  \/\ \    /\__  _/\ \/ /      /\  ___\/\  __ \/\ "-.\ \/\__  _/\  __ \/\ \/\ "-.\ \/\  ___\/\  == \   '
echo '\ \ \-./\ \ \ \___\/_/\ \\\ \  _"-.    \ \ \___\ \ \/\ \ \ \-.  \/_/\ \\\ \  __ \ \ \ \ \-.  \ \  __\\\ \  __<   '
echo ' \ \_\ \ \_\ \_____\ \ \_\\\ \_\ \_\    \ \_____\ \_____\ \_\\\"\_\ \ \_\\\ \_\ \_\ \_\ \_\\\"\_\ \_____\ \_\ \_\ '
echo '  \/_/  \/_/\/_____/  \/_/ \/_/\/_/     \/_____/\/_____/\/_/ \/_/  \/_/ \/_/\/_/\/_/\/_/ \/_/\/_____/\/_/ /_/ '
echo "_____________________________________________________________________________________________________________"
echo "Splunk> MLTK Container for TensorFlow 2.0, PyTorch and Jupyterlab."
tag="golden-image-cpu"
base="ubuntu:20.04"
dockerfile="Dockerfile"
repo="phdrieger/"
if [ -z "$1" ]; then
  echo "No build parameters set. Using default tag golden-image-cpu for building and running the container."
  echo "You can use ./build.sh [golden-image-cpu|golden-image-gpu|tf-cpu|tf-gpu|pytorch|nlp] to build the container for different frameworks."
else
  tag="$1"
fi
case $tag in
	template-cpu)
		base="python:3.9.13-bullseye"
		dockerfile="Dockerfile.5.0.0.minimal.cpu.template"
		;;
	template-gpu)
		base="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
		dockerfile="Dockerfile.5.0.0.minimal.gpu.template"
		;;
	golden-image-cpu)
		base="python:3.9.13-bullseye"
		dockerfile="Dockerfile.5.0.0.cpu"
		;;
	golden-image-gpu)
		base="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
		dockerfile="Dockerfile.5.0.0.gpu"
		;;		
	river)
		base="python:3.9"
		dockerfile="Dockerfile.5.0.0.river"
		;;
	spark)
		base="jupyter/all-spark-notebook:spark-3.2.1"
		dockerfile="Dockerfile.5.0.0.spark"
		;;
	rapids)	
		#base="rapidsai/rapidsai-core:22.04-cuda11.5-runtime-ubuntu20.04-py3.8"
		base="rapidsai/rapidsai-core:21.12-cuda11.0-runtime-ubuntu20.04-py3.8"
		dockerfile="Dockerfile.5.0.0.rapids"
		;;
	golden-image-gpu-3-9)
		base="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
		#base="nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04"
		dockerfile="Dockerfile.3.9.gpu"
		;;		
	river-3-9)
		base="python:3.9"
		dockerfile="Dockerfile.3.9.river"
		;;
	golden-image-cpu-3-9)
		base="ubuntu:20.04"
		dockerfile="Dockerfile.3.9.cpu"
		;;
	spark-3-9)
		base="jupyter/all-spark-notebook:spark-3.2.1"
		dockerfile="Dockerfile.3.9.spark"
		;;
	rapids-3-9)	
		base="rapidsai/rapidsai-core:21.12-cuda11.0-runtime-ubuntu20.04-py3.7"
		dockerfile="Dockerfile.3.9.rapids"
		;;
	transformers-cpu)
		#base="tensorflow/tensorflow:2.8.0"
		#dockerfile="Dockerfile.3.9.transformers.cpu"
		base="python:3.9.13-bullseye"
		dockerfile="Dockerfile.5.1.0.transformers.cpu"
		;;
	transformers-gpu)
		base="nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04"
		dockerfile="Dockerfile.5.1.0.transformers.gpu"
		;;
	*)
		echo "Invalid container image tag: $tag, expected [golden-image-cpu|golden-image-gpu|tf-cpu|tf-gpu|pytorch|pytorch-nlp]"
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
