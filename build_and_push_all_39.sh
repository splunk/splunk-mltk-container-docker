#!/bin/sh
echo "Build and push all 3.9 images"

./build.sh golden-image-cpu phdrieger/ 3.9.0
docker push phdrieger/mltk-container-golden-image-cpu:3.9.0

./build.sh spark phdrieger/ 3.9.0
docker push phdrieger/mltk-container-spark:3.9.0

./build.sh river phdrieger/ 3.9.0
docker push phdrieger/mltk-container-river:3.9.0

./build.sh golden-image-gpu phdrieger/ 3.9.0
docker push phdrieger/mltk-container-golden-image-gpu:3.9.0

#./build.sh rapids phdrieger/ 3.9.0
#docker push phdrieger/mltk-container-rapids:3.9.0
