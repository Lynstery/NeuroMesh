bash archiconda.sh
source ~/archiconda3/etc/profile.d/conda.sh
conda create --name mesh python=3.6
conda activate mesh
bash jetson_pytorch.sh
conda install -c conda-forge opencv
sudo apt-get install screen expect ffmpeg sshpass jq -y
pip3 install torchmetrics pytorch-msssim pycocotools joblib
pip3 install -r ~/workspace/mesh/baselines/crucio/dnn_model/yolov5_requirements.txt




