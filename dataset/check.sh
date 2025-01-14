#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
DATA_DISK_DIR=$(./data_disk.sh)
TRAIN_DIR=$DATA_DISK_DIR/kinetics-dataset/k400/train
TEST_DIR=$DATA_DISK_DIR/kinetics-dataset/k400/test
VAL_DIR=$DATA_DISK_DIR/kinetics-dataset/k400/val
REPLACE_DIR=$DATA_DISK_DIR/kinetics-dataset/k400/replacement/replacement_for_corrupted_k400
CORRUPTED_MP4=$DATA_DISK_DIR/kinetics-dataset/k400/corrupted_mp4.txt
rm -rf $CORRUPTED_MP4
touch $CORRUPTED_MP4

check_file() {
    filepath=$(realpath "$1")
    if ffmpeg -v error -i "$filepath" -f null - 2>&1 | grep -q "Invalid data found when processing input"; then
        # echo "File is corrupted: $filepath"
        rm -rf "$filepath"
        echo "$filepath" >>"$CORRUPTED_MP4"
    fi
}

export -f check_file
export CORRUPTED_MP4

echo "[ffmpeg] checking $TRAIN_DIR"
find "$TRAIN_DIR" -type f -name "*.mp4" | parallel --bar check_file {}
wait
echo "[ffmpeg] checking $TEST_DIR"
find "$TEST_DIR" -type f -name "*.mp4" | parallel --bar check_file {}
wait
echo "[ffmpeg] checking $VAL_DIR"
find "$VAL_DIR" -type f -name "*.mp4" | parallel --bar check_file {}
wait
echo "[ffmpeg] checking $REPLACE_DIR"
find "$REPLACE_DIR" -type f -name "*.mp4" | parallel --bar check_file {}
wait
