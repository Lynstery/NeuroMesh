#!/bin/bash

root="/data/wxk/AIC20_track3_MTMC/videos"

cd "$(dirname "$0")"

cd "../GT/StrongSORTYOLO"

for data in "$root/Untiled_mp4/"*; do
    if [ -d "$data" ]; then
        echo "Running DNN on $data."
        videoname=$(basename "$data")
        python detectTiles_StrongSORT.py --source "$data" --save-txt \
            --tiled-video "$root/tiled_4x4_mp4/$videoname/output0000_tiled.mp4" \
            --classes 0 1 2 3 4 5 6 7 \
            --save-labelfolder-name "../../../assets/GroundTruths_TileLevel/" \
            --yolo-weight "weights/yolov5n.pt" \
            --save-vid
        echo "Done for $data"
    fi
done