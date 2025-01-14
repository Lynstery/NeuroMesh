#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
input_path=$(./download.sh)
echo ${input_path}

if [ ! -d "${input_path}" ]; then
    echo "Directory ${input_path} does not exist"
    exit
fi
find ${input_path}/val2017 -name "*.pkl" -type f -delete
find ${input_path}/val2017 -name "*.mp4" -type f -delete
find ${input_path}/val2017 -name "*_rec.png" -type f -delete
find ${input_path}/youtube-objects -name "*.pkl" -type f -delete
find ${input_path}/youtube-objects -name "*.txt" -type f -delete
find ${input_path}/youtube-objects -name "*_rec" -type d -exec rm -r {} +
find ${input_path}/youtube-objects -name "*_n*" -type d -exec rm -r {} +
