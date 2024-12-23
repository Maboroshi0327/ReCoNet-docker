#!/bin/bash

echo -n "Datasets mount path: "
read -r DATASETS_PATH
echo -n "Image tag: "
read -r tag

docker create --name reconet --ipc host -it --gpus all \
    -v $DATASETS_PATH:/root/datasets \
    -v ./ReCoNet-PyTorch:/root/ReCoNet \
    $tag