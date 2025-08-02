#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {huggingface|vllm}"
  exit 1
fi

case "$1" in
    huggingface)
        # Within a Python 3.10 environment (kimivl2506); ensure that Cuda12.8 is being used
        #source ~/.bashrc
        #source switch-cuda.sh 12.8
        #pyenv activate kimivl2506
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # torch 2.7
        pip install transformers==4.48.2 
        wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
        pip install flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
        pip install tiktoken blobfile accelerate
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.7.0%2Bcu128.html
        ;;
    vllm)
        pip install vllm
        ;;
esac