import os
import ffmpeg
import csv
LUMPI_DIR='/data/zh/LUMPI-dataset'

def match_clip_pairs(measurementid, camid1, camid2):
    clips = []
    for video_name in os.listdir(os.path.join(LUMPI_DIR, f"Measurement{measurementid}/cam/{camid1}/clips")): 
        if video_name.endswith(".mp4"):
            video_path = os.path.join(LUMPI_DIR, f"Measurement{measurementid}/cam/{camid1}/clips", video_name)
            ref_video_name = video_name.replace(f"{camid1}_", f"{camid2}_")
            ref_video_path = os.path.join(LUMPI_DIR, f"Measurement{measurementid}/cam/{camid2}/clips", ref_video_name)
            if not os.path.exists(ref_video_path):
                continue
            probe = ffmpeg.probe(video_path)
            duration = int(probe['streams'][0]['nb_frames'])
            ref_probe = ffmpeg.probe(ref_video_path)
            ref_duration = int(ref_probe['streams'][0]['nb_frames'])
            if duration != ref_duration:
                continue
            clips.append((video_path, ref_video_path))
            print(f"{video_path} {ref_video_path}")
    return clips

clips = []
clips.extend(match_clip_pairs(5, 5, 6))
clips.extend(match_clip_pairs(5, 6, 7))
clips.extend(match_clip_pairs(5, 7, 5))

#clips.extend(match_clip_pairs(6, 6, 7))
#clips.extend(match_clip_pairs(6, 5, 6))
#clips.extend(match_clip_pairs(6, 7, 5))

csv_path = LUMPI_DIR + '/clips_pair.csv' 

with open(csv_path, 'w') as f_csv:
    writer = csv.writer(f_csv, delimiter=' ', lineterminator='\n',)
    for video_path, ref_video_path in clips:
        writer.writerow([video_path, ref_video_path])

f_csv.close()
