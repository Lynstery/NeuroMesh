#!/bin/bash

root="/data/wxk/AIC20_track3_MTMC/videos"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$root"
python $script_dir/../../utils/aggrTiles.py