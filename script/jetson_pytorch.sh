# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
python_dir=$(python3 -c "import sys; print(sys.prefix)")
echo "python dir: $python_dir"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_version=$(echo "$python_version" | cut -d '.' -f 1,2)
echo "version: $python_version"
jet_version=$(jetson_release | grep -oP 'Jetpack\s+\K[^\s]+' | sed 's/[^0-9.]//g')
jet_version="${jet_version:1:-1}"
echo "jet_version: $jet_version"

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev sysstat libsensors-config -y
pip3 install pysocks
pip3 install --upgrade pip

if [ "$jet_version" = "5.0" ]; then
    py_version=3.8
    # PyTorch v1.13.0
    torch_url="https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl"
    torch_aarch64="torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl"
    vision_version=0.13.0
elif [ "$jet_version" = "5.1.1" ] || [ "$jet_version" = "5.1.2" ]; then
    py_version=3.8
    # PyTorch v2.1.0
    torch_url="https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    torch_aarch64="torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    vision_version=0.16.1
elif [ "$jet_version" = "4.6.1" ]; then
    torch_url="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
    torch_aarch64="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    py_version=3.6
    vision_version=0.11.1
else
    echo "暂不支持$jet_version"
    exit
fi
if [ "$python_version" != "$py_version" ]; then
    echo "当前环境 python 版本为 $python_version, 需要 $py_version"
    exit
fi

# pytorch
echo "import torch;print(\"CUDA\",torch.cuda.is_available(),torch.version.cuda)" | python3 >/dev/null 2>&1
if [ $? -ne 0 ]; then
    curl -Lk $torch_url -o $torch_aarch64
    pip3 install 'Cython<3'
    pip3 install numpy --force-reinstall $torch_aarch64
    rm -rf $torch_aarch64
fi

# torchvision
echo "import torchvision;" | python3 >/dev/null 2>&1
if [ $? -ne 0 ]; then
    sudo apt-get install libpng-dev libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev -y
    git clone --branch v${vision_version} https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=${vision_version}
    python3 setup.py install --prefix "$python_dir"
    cd ../
    pip3 install 'pillow<7'
    rm -rf torchvision
fi

pip3 install timm einops ffmpeg-python tensorboardX pillow
pip3 install urllib3==1.26.6
sudo apt install parallel ffmpeg -y
echo
python3 -V
echo "import torch;print(\"Pytorch \"+torch.__version__)" | python3
echo "import torch;print(\"CUDA\",torch.cuda.is_available(),torch.version.cuda)" | python3
echo "import torchvision;import timm;import einops;import ffmpeg;import tensorboardX;import PIL" | python3