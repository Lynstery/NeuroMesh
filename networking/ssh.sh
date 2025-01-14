#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
CONFIGS=($(ls configs/*.json | xargs -n 1 basename | sed 's/.json$//'))
CONFIG_NUM=${#CONFIGS[@]}
hint="usage：./ssh.sh 或 ./ssh.sh <server id>"
if [ ! -n "$1" ]; then
    echo "choose a server:"
    let i=1
    for CONFIG in "${CONFIGS[@]}"; do
        echo "$i) ${CONFIG} $(cat configs/${CONFIG}.json | jq -r '.info')"
        let i=i+1
    done
    read -p "input number (1-${CONFIG_NUM}): " CHOICE
elif [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "-h" ] || [ "$1" == "-H" ]; then
        echo $hint
        exit
    else
        CHOICE="$1"
    fi
else
    echo $hint
    exit
fi
if ! [[ "$CHOICE" =~ ^[1-9][0-9]*$ ]] || [ "$CHOICE" -gt "$CONFIG_NUM" ]; then
    echo "invalid input，please input 1-${CONFIG_NUM}."
    exit
fi
CONFIG="./configs/${CONFIGS[$((CHOICE - 1))]}.json"
echo "use $CONFIG"
rm -rf config.json
cp $CONFIG ./config.json

HOST=$(cat config.json | jq -r '.host')
USERNAME=$(cat config.json | jq -r '.username')
PASSWORD=$(cat config.json | jq -r '.password')
IP=$(./get_ip.sh)
if [ "$IP" == "$HOST" ]; then
    echo -e "\e[31mNote: Execute this script locally!\e[0m"
    exit
fi

echo "Password is $PASSWORD"
ssh -Y $USERNAME@$HOST
