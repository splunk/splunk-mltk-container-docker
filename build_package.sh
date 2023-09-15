#!/bin/bash
# build dsdl-support package

if [ -f "./package-dsdlsupport/dist/dsdlsupport-1.0.0.tar.gz" ]; then
    echo "Support package already built, skipping build"
else        
    if [ -d "./package-dsdlsupport/venv" ]; then
        echo "./package-dsdlsupport/venv exists. Activating python venv for package building"
        source ./package-dsdlsupport/venv/bin/activate
    else
        echo "Configure a new python venv for package building in ./package-dsdlsupport/venv"
        python3 -m venv ./package-dsdlsupport/venv
        source ./package-dsdlsupport/venv/bin/activate
        pip3 install build
    fi
    echo "Building Python Package"
    cd ./package-dsdlsupport
    python3 -m build
    cd ..
fi
