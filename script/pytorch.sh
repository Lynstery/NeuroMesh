#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
error=1
#* https://pytorch.org/get-started/locally/
#* 2.3.0
# MIRRORS="http://mirrors.aliyun.com/pypi/simple/"
# HOST="mirrors.aliyun.com"
MIRRORS="http://mirrors.nju.edu.cn/pypi/web/simple/"
HOST="mirrors.nju.edu.cn"
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" = "1" ]; then
        error=0
        echo "install PyTorch..."
        pip3 install torch torchvision torchaudio -i $MIRRORS --trusted-host $HOST
        pip3 install --upgrade torch torchvision torchaudio -i $MIRRORS --trusted-host $HOST
        echo
        pip3 show torch
        cuda=$(grep -m 1 'cuda_ver=' './cuda.sh' | grep -oP 'cuda_ver\s*=\s*\K[^,]*')
        echo "recommended cuda version:"$cuda
    elif [ "$1" = "0" ]; then
        error=0
        echo "uninstall PyTorch..."
        pip3 uninstall torch torchvision torchaudio -y
    fi
fi
if [ $error -eq 1 ]; then
    echo "./pytorch.sh 1 install PyTorch"
    echo "./pytorch.sh 0 uninstall PyTorch"
fi
