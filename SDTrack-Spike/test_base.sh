# CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py SDTrack SDTrack-base --dataset eotb --threads 32 --num_gpus 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py SDTrack SDTrack-base --dataset visevent --threads 32 --num_gpus 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py SDTrack SDTrack-base --dataset coesot --threads 32 --num_gpus 4
