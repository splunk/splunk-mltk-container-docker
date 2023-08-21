#!/bin/sh
# build dsdl-support package

if [ -f "./package-dsdlsupport/dist/dsdlsupport-1.0.0.tar.gz" ]; then
    echo "Support package already built, skipping build"
else        
    if [ -d "./package-dsdlsupport/venv" ]; then
        source ./package-dsdlsupport/venv/bin/activate
    else
        python3 -m venv ./package-dsdlsupport/venv
        source ./package-dsdlsupport/venv/bin/activate
        pip3 install build
    fi
    cd ./package-dsdlsupport
    python3 -m build
fi