conda create -n SDTrack python=3.8
conda activate SDTrack
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

pip install thop
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom
pip install tensorboard
echo "****************** Downgrade setuptools ******************"
pip install setuptools==59.5.0


echo ""
echo ""
echo "****************** Installing wandb ******************"
pip install wandb

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm
pip install torchinfo
pip install einops
pip install fvcore
