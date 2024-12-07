FROM ubuntu:24.04

RUN apt update && apt upgrade && apt install -y wget \
    # Create directories
    && mkdir -p ~/ReCoNet \
    && mkdir -p ~/datasets \
    # Install Miniconda
    && mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh \
    # Install PyTorch
    && conda install -y python=3.8 \
    && conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
