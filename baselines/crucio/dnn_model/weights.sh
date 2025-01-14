#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
input_path=$(../video_data/download.sh)
echo ${input_path}

target_path="${HOME}/.cache/torch/hub"
if [ ! -d "${input_path}/checkpoints" ]; then
    echo "${input_path}/checkpoints dose not exist"
    exit
fi
if [ ! -f "${input_path}/ultralytics_yolov5_master.zip" ]; then
    echo "${input_path}/ultralytics_yolov5_master.zip dose not exist"
    exit
fi
if [ ! -d "${target_path}" ]; then
    echo "${target_path} dose not exist"
    exit
fi
if [ -d "${target_path}/checkpoints" ]; then
    rm -rf "${target_path}/checkpoints"
fi
cp -r "${input_path}/checkpoints" ${target_path}
if [ -d "${target_path}/ultralytics_yolov5_master" ]; then
    rm -rf "${target_path}/ultralytics_yolov5_master"
fi
if [ ! -d "${input_path}/ultralytics_yolov5_master" ]; then
    cd ${input_path}
    unzip ultralytics_yolov5_master.zip
fi
mv "${input_path}/ultralytics_yolov5_master" "${target_path}/ultralytics_yolov5_master"
