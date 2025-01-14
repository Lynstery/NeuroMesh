from pathlib import Path
import subprocess as sp
import argparse
import os
import shutil
import tempfile
import time

username = os.getlogin()
python_strongsortyolo = f"/data/zh/conda_env/strongsortyolo/bin/python"
python_tileclipper = f"/data/zh/conda_env/mesh/bin/python"

def Segment_Raw(imgs_path, num_frames, res):
    if os.path.exists(os.path.join(imgs_path, "segmented")):
        os.system(f'rm -rf {imgs_path}/segmented/*')
    else:
        os.makedirs(os.path.join(imgs_path, "segmented"))

    for i in range(num_frames//16):
        sp.run(f"ffmpeg -framerate 32 -start_number {i * 16 + 1} -i {imgs_path}/%06d.png -vframes 16 -pix_fmt yuv420p -s {res} {imgs_path}/segmented/seg{i:04d}.yuv".split(' '))

def Generate_Tiled_Segments(imgs_path, res, tiles):
    if os.path.exists(os.path.join(imgs_path, "tiled")):
        os.system(f'rm -rf {imgs_path}/tiled/*')
    else:
        os.makedirs(os.path.join(imgs_path, "tiled"))

    for idx, seg in enumerate(sorted(os.listdir(os.path.join(imgs_path, "segmented")))):
        seg_path = os.path.join(imgs_path, "segmented", seg)
        seg_name = os.path.splitext(seg)[0]
        with tempfile.NamedTemporaryFile(suffix='.hevc') as tmp_file:
            tmp_file_path = tmp_file.name 
            sp.run(f"kvazaar -i {seg_path} --input-res {res} --input-fps 32 --qp 30 --tiles {tiles} --slice tiles --mv-constraint frametilemargin -o {tmp_file_path}".split(" "))
            sp.run(f"MP4Box -add {tmp_file_path}:split_tiles -new {os.path.join(imgs_path, 'tiled')}/{seg_name}.mp4".split(" "))

def Generate_Untiled_Segments(imgs_path, res):
    if os.path.exists(os.path.join(imgs_path, "untiled")):
        os.system(f'rm -rf {imgs_path}/untiled/*')
    else:
        os.makedirs(os.path.join(imgs_path, "untiled"))

    for idx, seg in enumerate(sorted(os.listdir(os.path.join(imgs_path, "segmented")))):
        seg_path = os.path.join(imgs_path, "segmented", seg)
        seg_name = os.path.splitext(seg)[0]

        with tempfile.NamedTemporaryFile(suffix='.hevc') as tmp_file:
            tmp_file_path = tmp_file.name 
            sp.run(f"kvazaar -i {seg_path} --input-res {res} --input-fps 32 --qp 30 -o {tmp_file_path}".split(" "))
            sp.run(f"MP4Box -add {tmp_file_path} -new {os.path.join(imgs_path, 'untiled')}/{seg_name}.mp4".split(" "))
  
def run_dnn(imgs_path, save_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, "GT/StrongSORTYOLO"))
    untiled_path = os.path.join(imgs_path, "untiled")
    os.makedirs('../../../assets/GroundTruths_TileLevel/', exist_ok=True)
    sp.run([
        python_strongsortyolo, "detectTiles_StrongSORT.py",
        "--source", untiled_path,
        "--save-txt",
        "--tiled-video", f"{imgs_path}/tiled/seg0000.mp4",
        "--classes", "0", "1", "2", "3", "4", "5", "6", "7",
        "--save-labelfolder-name", f"../../../assets/GroundTruths_TileLevel/{save_name}.txt",
        "--yolo-weight", "../../weights/yolov5n.pt"
    ])

def run_calibrate(imgs_path, save_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sp.run([
        python_tileclipper, "calibrate.py",
        "--save-name", f"{save_name}",
        "--tiled-video-dir", f"{imgs_path}/tiled",
        "--assets-folder", "../assets",
        "--num-cal-seg", "30"
    ])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=str, default='/data/zh/videos_dir', help="videos dir")
    parser.add_argument("--num_frames", type=int, default=480, help="Number of frames")
    parser.add_argument("--res", type=str, default="224x224", help="Video Resolution, e.g. 1280x720")
    parser.add_argument("--tiles", type=str, default="4x4", help="Number of tiles to keep, e.g., 4x4")
    opt = parser.parse_args()
    
    res = opt.res
    tiles = opt.tiles
    num_frames = int(opt.num_frames)

    for idx, video_name in enumerate(sorted(os.listdir(opt.videos_dir))):
        video_path = os.path.join(opt.videos_dir, video_name)
        print(f"{video_path}")
        time.sleep(2)
        Segment_Raw(video_path, num_frames, res)
        Generate_Tiled_Segments(video_path, res, tiles)
        Generate_Untiled_Segments(video_path, res)
        run_dnn(video_path, video_name)
        run_calibrate(video_path, video_name)