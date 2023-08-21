#!/bin/sh
echo "Pull all DSDL 5.0.0 images"
docker pull phdrieger/mltk-container-rapids:5.0.0
docker pull phdrieger/mltk-container-template-cpu:5.0.0
docker pull phdrieger/mltk-container-golden-image-cpu:5.0.0
docker pull phdrieger/mltk-container-template-gpu:5.0.0
docker pull phdrieger/mltk-container-golden-image-gpu:5.0.0
docker pull phdrieger/mltk-container-spark:5.0.0
