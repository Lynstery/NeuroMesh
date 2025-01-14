#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
input_path="$(./download.sh)/youtube-objects"
echo ${input_path}

if [ ! -d "${input_path}" ]; then
    echo "Directory ${input_path} does not exist"
    exit
fi
# Maximum number of video frames (starting from 1)
MAX_FRAME_NUM=45
# https://data.vision.ee.ethz.ch/cvl/youtube-objects/
FILES=("aeroplane.tar.gz" "bird.tar.gz" "boat.tar.gz" "car.tar.gz" "cat.tar.gz" "cow.tar.gz" "dog.tar.gz" "horse.tar.gz" "motorbike.tar.gz" "train.tar.gz")
img_ext=".jpg"
# Cycle through each compressed file
for file in "${FILES[@]}"; do
    # Check whether compressed file exists
    if [ -f "${input_path}/${file}" ]; then
        # Extract file name
        filename=$(basename -- "$file")
        # Extract file name prefix
        prefix="${filename%%.*}"
        # Determine whether images for compressed file has been processed
        if ls "$input_path"/*${prefix}_* >/dev/null 2>&1; then
            echo "Skip ${prefix}"
        else
            echo "Process ${prefix}"
            # Check whether folder with same name exists
            if [ ! -d "${input_path}/${prefix}" ]; then
                # Unzip file
                tar -zxf "${input_path}/${file}" -C "${input_path}"
            fi
            # Process each subfolder
            if [ -d "${input_path}/${prefix}/data" ]; then
                for dir in "${input_path}/${prefix}/data/"*/; do
                    # Determines if subfolder is named with a number and
                    # directory name does not contain lourdes
                    if [[ ${dir} =~ [0-9] && ${dir} != *"lourdes"* ]]; then
                        # Gets subfolder name
                        subdir=$(basename "$dir")
                        # Process each second-level subfolder in subfolder
                        if [ ! -z "$(ls -A ${dir})" ]; then
                            for subsubdir in "${dir}shots/"*/; do
                                # Counts number of images in each second-level subfolder
                                image_number=$(find ${subsubdir} -maxdepth 1 -type f -name "*${img_ext}" | wc -l)
                                if [ "$image_number" -ge $MAX_FRAME_NUM ]; then
                                    echo $subsubdir
                                    # Gets second-level subfolder name
                                    subsubdir_name=$(basename "$subsubdir")
                                    # Move second-level subfolder and rename it
                                    video_path="${input_path}/${prefix}_${subdir}_${subsubdir_name}"
                                    if [ ! -d "${video_path}" ]; then
                                        mv "${subsubdir}" "${video_path}"
                                    fi
                                    # Delete redundant images and other files other than images
                                    find ${video_path} -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;
                                    find ${video_path} -maxdepth 1 -type f ! -name "*${img_ext}" -delete
                                    find ${video_path} -maxdepth 1 -type f -name "frame*${img_ext}" | while read filename; do
                                        number=$(echo $filename | grep -o '[0-9]\+' | tail -1)
                                        if [ "$number" -gt $MAX_FRAME_NUM ]; then
                                            rm -f $filename
                                        else
                                            file_basename=$(basename "$filename")
                                            num_digits=$(echo "$file_basename" | grep -o "[0-9]" | wc -l)
                                            # If image file number is more than 4 digits
                                            if [ "$num_digits" -gt 4 ]; then
                                                new_filename="frame${number: -4}${img_ext}"
                                                mv "$filename" "${video_path}/${new_filename}"
                                            fi
                                        fi
                                    done
                                fi
                            done
                        fi
                    fi
                done
                # Delete decompression directory
                rm -r "${input_path}/${prefix}"
            else
                echo "Directory ${input_path}/${prefix}/data does not exist"
            fi
        fi
    fi
done
find ${input_path} -type f -exec chmod -x {} \;
