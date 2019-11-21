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
if [ -z "$1" ]; then
  echo "No build parameters set. Using default tag tf-cpu for building and running the container."
  echo "You can use ./run.sh [tf-cpu|tf-gpu|pytorch] to run the container for different frameworks."
else
  tag="$1"
  echo "Using tag $tag for building and running the container"
fi
echo "Stop and remove container..."
docker stop mltk-container-$tag
docker rm mltk-container-$tag
echo "Starting container..."
docker run -it --rm --name mltk-container-$tag -l mltk_container -l mltk_model -p 5000:5000 -p 8888:8888 -p 6006:6006 -v mltk-container-app:/srv/app -v mltk-container-notebooks:/srv/notebooks mltk-container-$tag:latest