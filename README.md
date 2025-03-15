# SDTrack: A Baseline for Event-based Tracking via Spiking Neural Networks

## Conda Environment
You can download the compressed package of the environment [here](https://drive.google.com/file/d/1bHu7CbM6TiSXNXnMbfj8W-eUNvO_4wyA/view?usp=sharing).
or Or manually install the environment using `requirement.yml`.
## Data Prepare

## Download the pre-trained weights from ImageNet-1K.
1. Download [SDTrack-Tiny]() and [SDTrack-Base]().
2. Create the directory SDTrack/**pretrained_models** and place the two downloaded weight files into this directory.

## Modify the settings required for training and testing.
1. The training path configuration file is located at `SDTrack/lib/train/admin/local.py`.
2. The testing path configuration file is located at `SDTrack/lib/test/evaluation/local.py`.

## Training FE108
```
bash train_tiny_fe108.sh
bash train_base_fe108.sh
```

## Training VisEvent
```
bash train_tiny_visevent.sh
bash train_base_visevent.sh
```

## Training FELT
```
bash train_tiny_felt.sh
bash train_base_felt.sh
```
