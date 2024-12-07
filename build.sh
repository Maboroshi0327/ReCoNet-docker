#!/bin/bash

echo "Tag name:"
read -r tag
docker build -t $tag .