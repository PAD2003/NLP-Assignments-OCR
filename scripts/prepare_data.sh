#!/bin/bash

if [ -d "data/OCR" ]; then
    echo "Data already exists."
else
    echo "Downloading data ..."
    cd data

    gdown 1yPWF1JflLSC7ZU2xTMLJBcvAUCtW6Day -O OCR.zip
    unzip OCR.zip

    cd ..
fi