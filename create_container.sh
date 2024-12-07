#!/bin/bash

DATASETS_PATH="/mnt/d/Datasets"
docker run --name reconet -it --gpus all -v $DATASETS_PATH:/root/datasets maboroshi327/reconet