#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
if ! conda env list | grep -q 'strongsortyolo'; then
    conda create --name strongsortyolo python=3.8
fi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate strongsortyolo
pip install -r StrongSORTYOLO/requirements.txt
env_path=$(conda env list | grep '*' | awk '{print $3}')
pkg_path="$env_path"/lib/python3.8/site-packages

if [ ! -d "${pkg_path}/ffmpeg" ]; then
    unzip ffmpeg_lib.zip -d $pkg_path
fi
