#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
curr_dir=$(pwd)
DATA_DISK_DIR=$(./data_disk.sh)
cd $DATA_DISK_DIR"/LUMPI-dataset"

duration=10  # Duration of each segment in seconds


cd Measurement5

# (573, 513) (810, 672) (237, 159)

# (474, 507) (942, 711) (468, 204)

# (538, 538) (1087, 855) (549, 317)

# Loop through each input file
for input_dir in cam/*/; do
  video_name="$(basename "$input_dir")"

  if [ "$video_name" == "5" ]; then
    x=573
    y=513
    w=237
    h=159
  fi

  if [ "$video_name" == "6" ]; then
    x=474
    y=507
    w=468
    h=204
  fi

  if [ "$video_name" == "7" ]; then
    x=538 
    y=538
    w=549
    h=317
  fi

  input_file="${input_dir}video.mp4"
  crop_file="${input_dir}video_crop.mp4"

  echo "$input_file to $crop_file with crop=$w:$h:$x:$y"
  ffmpeg -i "$input_file" -filter:v "crop=$w:$h:$x:$y" "$crop_file"

  output_dir="${input_dir}clips"
  mkdir -p "$output_dir"
  output_prefix="${output_dir}/$(basename "$input_dir")"

  echo "$crop_file"
  echo "$output_prefix"

  ffmpeg -i "$crop_file" -c:v libx264 -c:a aac -map 0 -segment_time "$duration" -g $(($duration * 30)) -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*${duration})" -f segment -reset_timestamps 1 "${output_prefix}_%04d.mp4"
done