#!/bin/bash

# Within a Python 3.11.11 environment (kimivl):
source ~/.bashrc
pyenv activate kimivl
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.1%2Bcu118.html
pip install transformers tiktoken blobfile accelerate
