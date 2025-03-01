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

server_dir=/data/$USERNAME
read -p "please input absolute path to transfer" source_dir
check=$(echo "$source_dir" | grep -F "${server_dir}")
if [ -z "$check" ]; then
    echo "non-exist USERNAME@$HOST:${server_dir}"
    exit
fi
dest_dir=$(echo "$source_dir" | sed "s#${server_dir}#${DATA_DISK_DIR}#g")
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -d $source_dir ]" &&
    sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ ! -f $source_dir ]"; then
    echo "non-exist ${USERNAME}@${HOST}:${source_dir}"
    exit
fi

parent_dir="${dest_dir%/*}"
if [ ! -d $parent_dir ]; then
    mkdir -p $parent_dir
fi
rm -rf $dest_dir

echo "Password is $PASSWORD"
if sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ -d $source_dir ]"; then
    scp -r $USERNAME@$HOST:$source_dir $dest_dir
elif sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no $USERNAME@$HOST "[ -f $source_dir ]"; then
    scp $USERNAME@$HOST:$source_dir $dest_dir
fi
