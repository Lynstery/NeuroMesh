#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
DATA_DISK_DIR=$(./data_disk.sh)
cd $DATA_DISK_DIR/kinetics-dataset/k400
rm -rf annotations k400_320p kinetics400 kinetics_400_labels.csv train.csv test.csv val.csv corrupted_mp4.txt
