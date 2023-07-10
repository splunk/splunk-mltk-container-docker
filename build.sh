#!/bin/sh
echo '__________________________________________________________________________________________________________________'
echo ' ________  ________  ___       ___  ___  ________   ___  __            ________  ________  ________  ___          '
echo '|\   ____\|\   __  \|\  \     |\  \|\  \|\   ___  \|\  \|\  \         |\   ___ \|\   ____\|\   ___ \|\  \         '
echo '\ \  \___|\ \  \|\  \ \  \    \ \  \\\  \ \  \\ \  \ \  \/  /|_       \ \  \_|\ \ \  \___|\ \  \_|\ \ \  \        '
echo ' \ \_____  \ \   ____\ \  \    \ \  \\\  \ \  \\ \  \ \   ___  \       \ \  \ \\ \ \_____  \ \  \ \\ \ \  \       '
echo '  \|____|\  \ \  \___|\ \  \____\ \  \\\  \ \  \\ \  \ \  \\ \  \       \ \  \_\\ \|____|\  \ \  \_\\ \ \  \____  '
echo '    ____\_\  \ \__\    \ \_______\ \_______\ \__\\ \__\ \__\\ \__\       \ \_______\____\_\  \ \_______\ \_______\'
echo '   |\_________\|__|     \|_______|\|_______|\|__| \|__|\|__| \|__|        \|_______|\_________\|_______|\|_______|'
echo '   \|_________|                                                                    \|_________|                   '
echo '__________________________________________________________________________________________________________________'
echo 'Splunk> DSDL Container Build Script for Custom Data-Science Runtimes'

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

line=$(grep "^${tag}," tag_mapping.csv)

if [ "$line" != "" ]; then
    base_image=$(echo $line | cut -d',' -f2)
    dockerfile=$(echo $line | cut -d',' -f3)
    base_requirements=$(echo $line | cut -d',' -f4)
    specific_requirements=$(echo $line | cut -d',' -f5)
    runtime=$(echo $line | cut -d',' -f6) 

    echo "Tag: $tag"
    echo "Base Image: $base_image"
    echo "Dockerfile: $dockerfile"
    echo "Base Requirements File: $base_requirements"
    echo "Specific Requirements File: $specific_requirements"
    echo "Runtime Options: $runtime"
else
    echo "No match found for tag: $tag"
    exit
fi

echo "Building custom module."
./package-dsdlsupport/build_package.sh

container_name="$repo"mltk-container-$tag
echo "Target container name: $container_name"

echo "Stopping and removing running containers with this name."
docker stop $container_name
docker rm $container_name
docker rmi $container_name

base_requirements_id="${base_requirements%.*}"
specific_requirements_id="${specific_requirements%.*}"

compiled_requirements_id=compiled_${base_requirements_id}_${specific_requirements_id}_$tag
compiled_requirements_filename=./requirements_files/$compiled_requirements_id.txt

echo "Checking for compiled requirements $compiled_requirements_id"

if [[ -f $compiled_requirements_filename ]]; then
  echo "Found pre-compiled requirements: Using $compiled_requirements_id instead of $base_requirements and $specific_requirements"
  base_requirements=$compiled_requirements_id.txt
  specific_requirements="empty".txt
fi

docker build --rm -t $container_name:$version\
  --build-arg BASE_IMAGE=$base_image \
  --build-arg TAG=$tag \
  --build-arg REQUIREMENTS_PYTHON_BASE=$base_requirements \
  --build-arg REQUIREMENTS_PYTHON_SPECIFIC=$specific_requirements \
  -f ./dockerfiles/$dockerfile \
  .

echo "Creating images.conf, move this file or copy-paste contents into <splunk_dir>/etc/apps/mltk-container/local/images.conf:"

# ensure both options are present if the nvidia runtime is specified
if [ "$runtime" = "nvidia" ]; then
  runtime="none,nvidia"
fi

echo -e "[$tag]\ntitle = $tag\nimage = mltk-container-$tag:$version\nrepo = $repo\nruntime = $runtime" > ./images_conf_files/$tag-images.txt

# Remove output file if it already exists
rm -f ./images_conf_files/images.conf

# Loop over all .txt files in the current directory
for file in ./images_conf_files/*-images.txt; do
  # Concatenate the file to the output file
  cat "$file" >> ./images_conf_files/images.conf
done