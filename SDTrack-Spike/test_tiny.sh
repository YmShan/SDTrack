CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py SDTrack SDTrack-tiny --dataset eotb --threads 32 --num_gpus 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py SDTrack SDTrack-tiny --dataset visevent --threads 32 --num_gpus 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test.py SDTrack SDTrack-tiny --dataset coesot --threads 32 --num_gpus 4