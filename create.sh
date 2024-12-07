#!/bin/bash

echo "Datasets mount path:"
read -r DATASETS_PATH
echo "Image tag:"
read -r tag

docker create --name reconet -it --gpus all \
    -v $DATASETS_PATH:/root/datasets \
    -v ./ReCoNet-PyTorch:/root/reconet \
    $tag