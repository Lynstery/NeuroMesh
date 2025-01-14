#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
curr_dir=$(pwd)
DATA_DISK_DIR=$(./data_disk.sh)
cd $DATA_DISK_DIR
# download LUMPI dataset 
# https://data.uni-hannover.de/dataset/lumpi
# https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/

# download only cameras in Mesurement5, Mesurement6

function download_file {
    wget --tries=5 --progress=bar -c -O "$1" "$2"
}
echo "Downloading LUMPI dataset to $DATA_DISK_DIR"

if [ ! -d "LUMPI-dataset" ]; then
    mkdir LUMPI-dataset
fi
cd LUMPI-dataset
if [ ! -d "test_data" ]; then
    download_file "test_data.zip" "https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/test_data.zip"
    unzip test_data.zip
fi

if [ ! -d "Label" ]; then
    download_file "labels.zip" "https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/labels.zip"
    unzip labels.zip
fi

download_file "meta.json" "https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/meta.json"

if [ ! -d "Measurement5" ]; then
    mkdir Measurement5
fi
cd Measurement5
if [ ! -d "background" ]; then
    download_file background.zip https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement5/background.zip
    unzip background.zip
fi
if [ ! -d "cam" ]; then
    download_file cam.zip.001 https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement5/cam.zip.001
    unzip cam.zip.001
fi

cd ..
if [ ! -d "Measurement6" ]; then
    mkdir Measurement6
fi
cd Measurement6
if [ ! -d "background" ]; then
    download_file background.zip.001 https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement6/background.zip.001
    unzip background.zip.001
fi
if [ ! -d "cam" ]; then
    download_file cam.zip.001 https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement6/cam.zip.001
    unzip cam.zip.001
fi

