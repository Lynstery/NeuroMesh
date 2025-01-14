#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
error=1
ARCH=$(uname -m)
release_num=$(lsb_release -r --short)
shell=$(env | grep SHELL=)
echo $shell
if [ $shell == "SHELL=/bin/bash" ]; then
    rc=.bashrc
else
    rc=.zshrc
fi
py_version=""
fix_server() {
    if grep -q ".proxychains4.conf" ${HOME}/$rc; then
        http_proxy_bak=$http_proxy
        https_proxy_bak=$https_proxy
        ftp_proxy_bak=$ftp_proxy
        unset http_proxy
        unset https_proxy
        unset ftp_proxy
        proxychains4 -f ${HOME}/.proxychains4.conf "$@"
        export http_proxy=$http_proxy_bak
        export https_proxy=$https_proxy_bak
        export ftp_proxy=$ftp_proxy_bak
    else
        "$@"
    fi
}
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ $1 == "1" ]; then
        error=0
        echo "安装环境..."
        if [ "$ARCH" = "x86_64" ]; then
            if command -v "nvidia-smi" >/dev/null 2>&1; then
                nvidia-smi
            else
                echo "[安装NVIDIA驱动] 软件和更新->附加驱动->专用,tested"
            fi
        elif [ "$ARCH" = "aarch64" ]; then
            jetson_info=$(jetson_release)
            t_model=194
            if echo "$jetson_info" | grep -q "AGX Orin"; then
                echo "NVIDIA Jetson AGX Orin"
                t_model=234
            elif echo "$jetson_info" | grep -q "Xavier NX"; then
                echo "NVIDIA Jetson Xavier NX"
            elif echo "$jetson_info" | grep -q "AGX Xavier"; then
                echo "NVIDIA Jetson AGX Xavier"
            else
                echo "Unknown device"
                exit
            fi
            # https://developer.nvidia.com/embedded/jetson-linux-archive
            model_version=$(jetson_release | grep -oP 'L4T\s+\K[^\s]+' | sed 's/[^0-9.]//g')
            model_version="${model_version:1:-1}"
            model_version=$(echo "$model_version" | cut -d '.' -f 1-2)
            nvidia_apt=nvidia-l4t-apt-source.list
            sudo rm -rf /etc/apt/sources.list.d/$nvidia_apt
            cat >$nvidia_apt <<EOF
deb https://repo.download.nvidia.com/jetson/common r${model_version} main
deb https://repo.download.nvidia.com/jetson/t${t_model} r${model_version} main
EOF
            sudo mv $nvidia_apt /etc/apt/sources.list.d/$nvidia_apt
            sudo apt-get update -y
            sudo apt dist-upgrade -y
            case "$release_num" in
            "20.04")
                libtbb2_arm64=libtbb2_arm64.deb
                curl http://launchpadlibrarian.net/463899398/libtbb2_2020.1-2_arm64.deb -o $libtbb2_arm64
                sudo dpkg -i $libtbb2_arm64
                sudo apt-get install nvidia-jetpack nvidia-jetpack-runtime nvidia-l4t-jetson-multimedia-api nvidia-l4t-camera nvidia-l4t-multimedia \
                    libglvnd-dev nvidia-jetpack-dev nvidia-nsight-sys nvidia-opencv-dev nvidia-opencv libopencv-dev libopencv-python libopencv-samples \
                    opencv-licenses libopencv nsight-systems-2022.5.2 libxkbcommon-x11-0=0.8.2-1~ubuntu18.04.1 libxkbcommon0=0.8.2-1~ubuntu18.04.1 \
                    nvidia-l4t-multimedia-utils libegl1 nvidia-l4t-core nvidia-l4t-nvsci libegl-mesa0 libgbm1=20.0.8-0ubuntu1~18.04.1
                rm -rf $libtbb2_arm64
                sudo apt-get install libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev libjpeg-turbo8=1.5.2-0ubuntu5.18.04.6 zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
                sudo apt-get install libavdevice57 libgl1 libglx0 libglvnd0=1.0.0-2ubuntu2.3 libgl1-mesa-dri libglapi-mesa=20.0.8-0ubuntu1~18.04.1
                ;;
            *)
                sudo apt-get install nvidia-jetpack -y
                sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev -y
                ;;
            esac
            fix_server python3 -m pip install pysocks
            # sudo rm -rf /usr/local/bin/jtop /usr/local/jtop;pip3 uninstall jetson-stats;pip3 cache purge;sudo reboot
            sudo -H pip3 install --no-cache-dir -U jetson-stats
            sudo systemctl restart jtop.service
            jet_version=$(jetson_release | grep -oP 'Jetpack\s+\K[^\s]+' | sed 's/[^0-9.]//g')
            jet_version="${jet_version:1:-1}"
            echo "$jet_version"
        fi
        if [ -d "${HOME}/anaconda3" ]; then
            anaconda -V
        elif [ -d "${HOME}/miniconda3" ]; then
            conda -V
        else
            curl --head --max-time 3 --silent --output /dev/null --fail "gitee.com"
            if [ $? -ne 0 ]; then
                echo "互联网连接存在问题"
                exit
            fi
            if [ "$ARCH" = "x86_64" ]; then
                # https://docs.anaconda.com/free/anaconda/install/linux/
                url="https://repo.anaconda.com/archive/"
                latest_version=$(curl -s "$url" | grep href | sed 's/.*href="//' | sed 's/".*//' | awk '/Linux-x86_64.sh/{print; exit}')
            elif [ "$ARCH" = "aarch64" ]; then
                url="https://repo.anaconda.com/miniconda/"
                latest_version="Miniconda3-latest-Linux-aarch64.sh"
            fi
            if [ ! -f "$latest_version" ]; then
                curl -O $url$latest_version
            fi
            chmod +x $latest_version
            echo "注意：面对\"Do you wish to update your shell profile to automatically initialize conda?\"时应选择yes"
            ./$latest_version
            rm -rf $latest_version
            echo "请在新终端中再次执行environment.sh"
            exit
        fi
        if [ "$ARCH" = "x86_64" ]; then
            # https://pytorch.org/get-started/locally/
            cuda_version=12.1
            small=0
            if conda list | grep -q "cuda"; then
                :
            else
                conda install nvidia/label/cuda-$cuda_version.$small::cuda -y
            fi
            if conda list | grep -q "pytorch"; then
                :
            else
                conda install pytorch torchvision torchaudio pytorch-cuda=$cuda_version -c pytorch -c nvidia -y
            fi
        elif [ "$ARCH" = "aarch64" ]; then
            sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev sysstat libsensors4 -y
            read -p "接下来需要GRACE"
            # https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
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
            else
                echo "暂不支持$jet_version"
                exit
            fi
            fix_server conda install python=$py_version
            fix_server pip3 install pysocks
            pip3 install --upgrade pip
            echo "import torch;print(\"CUDA\",torch.cuda.is_available(),torch.version.cuda)" | python3 >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                curl -Lk $torch_url -o $torch_aarch64
                pip3 install 'Cython<3'
                pip3 install numpy --force-reinstall $torch_aarch64
                rm -rf $torch_aarch64
                git clone --branch v${vision_version} https://github.com/pytorch/vision torchvision
                cd torchvision
                export BUILD_VERSION=${vision_version}
                python3 setup.py install --user
                cd ../
                pip3 install 'pillow<7'
                rm -rf torchvision
            fi
        fi
        pip3 install timm einops ffmpeg-python tensorboardX pillow
        sudo apt install parallel ffmpeg -y
        echo
        python3 -V
        echo "import torch;print(\"Pytorch \"+torch.__version__)" | python3
        echo "import torch;print(\"CUDA\",torch.cuda.is_available(),torch.version.cuda)" | python3
        echo "import torchvision;import timm;import einops;import ffmpeg;import tensorboardX;import PIL" | python3
        if [ "$py_version" ]; then
            echo -e "\e[32m所有环境仅在Python ${py_version}中有效\e[0m"
        else
            echo "在VSCode中可以切换右下角的Python解释器来使用虚拟环境"
        fi
    elif [ $1 == "0" ]; then
        error=0
        echo "卸载环境..."
        read -p "按回车键继续..."
        pip3 uninstall timm einops ffmpeg-python tensorboardX pillow torch torchvision -y
        pip3 cache purge
        sudo apt remove --purge parallel ffmpeg -y
        read -p "是否卸载Conda(Y/n)" isconda
        if [ "$isconda" == "Y" ] || [ "$isconda" == "y" ]; then
            cd ${HOME}/
            rm -rf anaconda3 miniconda3 .conda .condarc .continuum
            sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<<$/d' ${HOME}/.bashrc
            if [ -f "${HOME}/.zshrc" ]; then
                sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<<$/d' ${HOME}/.zshrc
            fi
        fi
    fi
fi
if [ $error == "1" ]; then
    echo "./environment.sh 1	安装环境"
    echo "./environment.sh 0	卸载环境"
    echo "1.若出现\"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver\"问题请通过[软件和更新->附加驱动]重新安装NVIDIA驱动"
    echo "2.若出现\"UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment\"问题请执行\"sudo apt-get install nvidia-modprobe -y\"后重启"
fi
