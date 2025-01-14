#!/bin/bash
#####################################################
# This script runs TileClipper on all videos
#####################################################

root="/data/wxk/AIC20_track3_MTMC/videos"
gamma=1.75
cd "$(dirname "$0")"

for data in $root/tiled_4x4_mp4/*; do
    if [ -d "$data" ]; then
        echo "Running tileclipper on $data."
        python ../tileClipper.py --tiled-video-dir "$data" --percentile-array-filename "../../assets/F2s/f2s_$(basename "$data")_cluster10.pkl" --cluster-indices-file "../../assets/F2s/$(basename "$data")_cluster_indices.pkl" --gamma "$gamma" 
        echo "Done for $data"
    fi
done