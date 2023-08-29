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
  tag_file="tag_mapping.csv"
  echo "No tag file specified. Using default tag_file: ${tag_file}"
else
  tag_file="$1"
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

# Read the first column values (excluding header) using awk
awk -F, 'NR>1 {print $1}' $tag_file | while read line; do
    # $line now contains the value from the first column

    echo "Building ${line}, see bulk_build.log for details."
    ./build.sh $line $repo $version >> bulk_build.log
done
