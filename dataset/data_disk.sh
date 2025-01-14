#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input

#HOST=$(cat ../networking/config.json | jq -r '.host')
#IP=$(../networking/get_ip.sh)
#if [ "$IP" == "$HOST" ]; then
#    DATA_DISK_DIR="/data/${USER}"
#else
#    MSI="/media/${USER}/PortableSSD"
#    if [[ -d "${MSI}" && -n "$(ls -A ${MSI})" ]]; then
#        DATA_DISK_DIR="${MSI}"
#    else
#        DATA_DISK_DIR="/media/${USER}/数据硬盘"
#    fi
#fi
DATA_DISK_DIR="/data/${USER}"
echo ${DATA_DISK_DIR}
