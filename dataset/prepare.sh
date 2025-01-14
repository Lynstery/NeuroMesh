#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
curr_dir=$(pwd)
DATA_DISK_DIR=$(./data_disk.sh)
cd $DATA_DISK_DIR
if [ ! -d "kinetics-dataset" ]; then
    git clone https://github.com/wgcban/kinetics-dataset.git
fi
cd kinetics-dataset
if [ ! -d "k400" ]; then
    if [ ! -d "k400_targz" ]; then
        chmod +x k400_downloader.sh
        ./k400_downloader.sh
    fi
    chmod +x k400_extractor.sh
    ./k400_extractor.sh
fi
cd k400
if [ ! -f "kinetics_400_labels.csv" ]; then
    curl -O https://gist.githubusercontent.com/willprice/f19da185c9c5f32847134b87c1960769/raw/9dc94028ecced572f302225c49fcdee2f3d748d8/kinetics_400_labels.csv
fi
if [ ! -d "kinetics400" ]; then
    if [ ! -f "kinetics400.tar.gz" ]; then
        curl -O https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz
    fi
    tar -xf 'kinetics400.tar.gz'
    rm -rf 'kinetics400.tar.gz'
fi

if [ ! -f "corrupted_mp4.txt" ]; then
    $curr_dir/check.sh
    python3 $curr_dir/check.py
fi
replacement_file() {
    file_path="$1"
    if [[ "$file_path" == *train* ]]; then
        dir_type="train"
    elif [[ "$file_path" == *test* ]]; then
        dir_type="test"
    elif [[ "$file_path" == *val* ]]; then
        dir_type="validate"
    else
        # echo "路径不包含train, test或val: $file_path"
        exit
    fi
    file_name=$(basename "$file_path")
    replacement_file_path="replacement/replacement_for_corrupted_k400/$file_name"
    if [ -f "$replacement_file_path" ]; then
        cp "$replacement_file_path" "$file_path"
    else
        # echo "${file_path}没有可替代视频"
        base_name="${file_name%.*}"
        youtube_id=$(echo "$base_name" | awk -F '_' '{print $1}')
        if [ ! -n "$youtube_id" ]; then
            youtube_id="_"$(echo "$base_name" | awk -F '_' '{print $2}')
        fi
        csv_file="kinetics400/${dir_type}.csv"
        sed -i "/${youtube_id}/d" "$csv_file"
    fi
}
if [ -s "corrupted_mp4.txt" ]; then
    export -f replacement_file
    echo "replacing corrupted mp4 files"
    cat corrupted_mp4.txt | parallel --bar replacement_file {}
    echo -n >corrupted_mp4.txt
fi

new_short_edge=320
#new_short_edge=480
process_mp4() {
    file="$1"
    new_short_edge="$2"
    filename=$(basename "$file")
    output_file="k400_${new_short_edge}p/"$filename
    ffmpeg -loglevel fatal -i "$file" -vf "scale=w=min(${new_short_edge}\,iw):-2" "$output_file"
    if [ $? -ne 0 ]; then
        # echo "Deleting $filename"
        rm -f "$file" "$output_file"
    fi
}
if [ ! -d "k400_${new_short_edge}p" ]; then
    mkdir k400_${new_short_edge}p
    export -f process_mp4
    echo "processing train"
    find "train" -type f -name "*.mp4" | parallel --bar process_mp4 {} "$new_short_edge"
    wait
    echo "processing test"
    find "test" -type f -name "*.mp4" | parallel --bar process_mp4 {} "$new_short_edge"
    wait
    echo "processing val"
    find "val" -type f -name "*.mp4" | parallel --bar process_mp4 {} "$new_short_edge"
    wait
fi
X=$(find k400_${new_short_edge}p -type f -name "*.mp4" | sed 's|^.*/||' | wc -l)
Y=$(find train test val -type f -name "*.mp4" | sed 's|^.*/||' | sort -u | wc -l)
if [ "$X" -eq "$Y" ]; then
    echo "OK:$X"
else
    echo "Mismatch: k400_${new_short_edge}p mp4 num=$X, mp4 num=$Y"
fi
