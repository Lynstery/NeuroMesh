import csv
import os

from mesh.dataset.utils import KINETICS400_DIR

# Set datapath and CSV save path
DATA_PATH = KINETICS400_DIR + '/k400_320p'

# get the class number
labels_400 = {}
with open(KINETICS400_DIR+"/kinetics_400_labels.csv") as f_labels:
    reader_labels = csv.reader(f_labels, delimiter="\t")
    next(reader_labels)
    for i, line in enumerate(reader_labels):
        n = line[0].split(",")[1]
        v = line[0].split(",")[0]
        labels_400[n] = v


# Generating "train" split
train_csv_path = os.path.join(KINETICS400_DIR, 'train.csv')
if os.path.exists(train_csv_path):
    os.remove(train_csv_path)
with open(train_csv_path, 'w') as f_csv:
    writer = csv.writer(f_csv, delimiter=' ', lineterminator='\n',)

    with open(KINETICS400_DIR+"/kinetics400/train.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(',')
            video_path = DATA_PATH+"/" + line[1]+"_" + \
                line[2].zfill(6)+"_"+line[3].zfill(6)+".mp4"
            if os.path.exists(video_path):
                video_lbl_num = labels_400[line[0]]
                video_duration = int(line[3]) - int(line[2])
                print(video_path, video_duration, video_lbl_num)
                writer.writerow([video_path, video_duration, video_lbl_num])
f_csv.close()
f.close()

# Generating "val" split
val_csv_path = os.path.join(KINETICS400_DIR, 'val.csv')
if os.path.exists(val_csv_path):
    os.remove(val_csv_path)
with open(val_csv_path, 'w') as f_csv:
    writer = csv.writer(f_csv, delimiter=' ', lineterminator='\n',)

    with open(KINETICS400_DIR+"/kinetics400/validate.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(',')
            video_path = DATA_PATH+"/" + line[1]+"_" + \
                line[2].zfill(6)+"_"+line[3].zfill(6)+".mp4"
            if os.path.exists(video_path):
                video_lbl_num = labels_400[line[0]]
                video_duration = int(line[3]) - int(line[2])
                print(video_path, video_duration, video_lbl_num)
                writer.writerow([video_path, video_duration, video_lbl_num])
f_csv.close()
f.close()

# Generating "train" split
test_csv_path = os.path.join(KINETICS400_DIR, 'test.csv')
if os.path.exists(test_csv_path):
    os.remove(test_csv_path)
with open(test_csv_path, 'w') as f_csv:
    writer = csv.writer(f_csv, delimiter=' ', lineterminator='\n',)

    with open(KINETICS400_DIR+"/kinetics400/test.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            line = line[0].split(',')
            video_path = DATA_PATH+"/" + line[1]+"_" + \
                line[2].zfill(6)+"_"+line[3].zfill(6)+".mp4"
            if os.path.exists(video_path):
                video_lbl_num = labels_400[line[0]]
                video_duration = int(line[3]) - int(line[2])
                print(video_path, video_duration, video_lbl_num)
                writer.writerow([video_path, video_duration, video_lbl_num])
f_csv.close()
f.close()
