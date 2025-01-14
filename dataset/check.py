import concurrent.futures
import os
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from mesh.dataset.utils import KINETICS400_DIR

directories = [
    Path(KINETICS400_DIR) / 'train',
    Path(KINETICS400_DIR) / 'test',
    Path(KINETICS400_DIR) / 'val',
    Path(KINETICS400_DIR) / 'replacement/replacement_for_corrupted_k400'
]
log_file = Path(KINETICS400_DIR) / 'corrupted_mp4.txt'


def check_and_delete_video(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        if 'streams' not in probe:
            raise Exception('No streams found')
        return None
    except Exception as e:
        os.remove(file_path)
        return file_path


def check_directory(directory):
    corrupted_files = []
    files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    for file in tqdm(files, desc=f'[ffmpeg] {directory}'):
        file_path = os.path.join(directory, file)
        result = check_and_delete_video(file_path)
        if result:
            corrupted_files.append(result)
    return corrupted_files


if __name__ == '__main__':
    if not os.path.exists(log_file):
        print('run check.sh first!')
        exit(1)
    all_corrupted_files = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_directory, directory)
                   for directory in directories]
        for future in concurrent.futures.as_completed(futures):
            all_corrupted_files.extend(future.result())
    with open(log_file, 'a') as f:
        for file_path in all_corrupted_files:
            f.write(file_path + '\n')
