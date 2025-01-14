#!/bin/bash
#####################################################
# This script runs TileClipper calibration on videos
#####################################################

root="/data/wxk/AIC20_track3_MTMC/videos"

cd "$(dirname "$0")"
for data in $root/tiled_4x4_mp4/*; do
    if [ -d "$data" ]; then
        echo "Running calibration on $data."
        python ../calibrate.py --tiled-video-dir "$data" --assets-folder ../../assets --num-cal-seg 60
        echo "Done for $data"
    fi
done