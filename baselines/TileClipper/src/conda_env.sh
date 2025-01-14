#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
. ~/anaconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q 'tileclipper'; then
    conda create --name tileclipper python=3.9 pip=23
fi
conda activate tileclipper
pip install -r requirements.txt
