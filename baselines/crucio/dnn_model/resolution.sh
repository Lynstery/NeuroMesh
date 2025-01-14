#!/bin/bash
input1=${HOME}/.local/lib/python3.*/site-packages/torchvision/models/detection
input2=${HOME}/.cache/torch/hub/ultralytics_yolov5_master/utils
input1=$(echo ${input1})
if [ ! -d "${input1}" ]; then
    echo "${input1}/ dose not exist"
    exit
fi
echo ${input1}
if [ ! -d "${input2}" ]; then
    echo "${input2}/ dose not exist"
    exit
fi
echo ${input2}
error=1
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        find=$(grep -n "if self.fixed_size:" ${input1}/transform.py | cut -d ":" -f 1)
        if [ -z "${find}" ]; then
            sed -i "s/image, target_index = self.resize(image, target_index)/\
if self.fixed_size:\n                image, target_index = self.resize(image, target_index)/g" ${input1}/transform.py
        fi
        echo "Modify PyTorch (transform.py)"
        sed -i "s/if new_size != imgsz:/if 1 == 0:/g" ${input2}/general.py
        echo "Modify YOLOv5 (general.py)"
    elif [ $1 == "0" ]; then
        error=0
        find=$(grep -n "if self.fixed_size:" ${input1}/transform.py | cut -d ":" -f 1)
        if [ "${find}" ]; then
            sed -i "${find}d" ${input1}/transform.py
            sed -i "s/    image, target_index = self.resize(image, target_index)/\
image, target_index = self.resize(image, target_index)/g" ${input1}/transform.py
        fi
        echo "Restore PyTorch (transform.py)"
        sed -i "s/if 1 == 0:/if new_size != imgsz:/g" ${input2}/general.py
        echo "Restore YOLOv5 (general.py)"
    fi
fi
if [ $error == "1" ]; then
    echo "./resolution.sh 1	Disallow DNN model scaling resolution"
    echo "./resolution.sh 0	Allow DNN models to scale resolution"
fi
