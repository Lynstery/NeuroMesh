#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
HOST=$(cat config.json | jq -r '.host')
USERNAME=$(cat config.json | jq -r '.username')
PASSWORD=$(cat config.json | jq -r '.password')
DATA_DISK_DIR=$(../dataset/data_disk.sh)
IP=$(./get_ip.sh)
if [ "$IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Execute this script locally!\e[0m"
    exit
fi

local_dir=${DATA_DISK_DIR}
read -p "请输入待传输目录/文件的绝对路径" source_dir
check=$(echo "$source_dir" | grep -F "${local_dir}")
if [ -z "$check" ]; then
    echo "将待传输目录/文件放在${local_dir}下"
    exit
fi
dest_dir=$(echo "$source_dir" | sed "s#${local_dir}#/data/$USERNAME#g")
if [ ! -d "$source_dir" ] && [ ! -f "$source_dir" ]; then
    echo "本地不存在路径为${source_dir}的目录/文件"
    exit
fi

parent_dir="${dest_dir%/*}"
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $parent_dir ]"; then
    sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "mkdir -p $parent_dir"
fi
sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "rm -rf $dest_dir"

echo "Password is $PASSWORD"
if [ -d "$source_dir" ]; then
    scp -r $source_dir $USERNAME@$HOST:$dest_dir
elif [ -f "$source_dir" ]; then
    scp $source_dir $USERNAME@$HOST:$dest_dir
fi
