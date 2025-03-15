# SDTrack: A Baseline for Event-based Tracking via Spiking Neural Networks

## Conda Environment
You can download the compressed package of the environment [here](https://drive.google.com/file/d/1bHu7CbM6TiSXNXnMbfj8W-eUNvO_4wyA/view?usp=sharing).
or Or manually install the environment using `requirement.yml`.
## Data Prepare

## Download the pre-trained weights from ImageNet-1K.
1. Download [SDTrack-Tiny](https://drive.google.com/file/d/1OcXHCnibEv9F40gw5VwGO90adtE6E0Ik/view?usp=sharing) and [SDTrack-Base](https://drive.google.com/file/d/1maJd0td46oxHACeBk2Vc90a__VyDAeWj/view?usp=sharing).
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

## Get the training and inference results.
### Weights
|  | FE108 | FELT | VisEvent |
|----------|----------|----------|----------|
| SDTrack-Tiny    |  [link](https://drive.google.com/file/d/1Hal0RcEgYKuqBiUFwPHa8f2bisboIp80/view?usp=sharing)  |  [link](https://drive.google.com/file/d/1MK2af7FG-TSHAUsw0eR4PQ7gmuEP34DP/view?usp=sharing)  | [link](https://drive.google.com/file/d/1rbZT2DBMeKrWZ8ORwNDz9fBKoMqRGN-_/view?usp=sharing)   |
| SDTrack-Base    | [link](https://drive.google.com/file/d/1tnJme3hugllA8xAIODoARzKaOkQKh6jr/view?usp=sharing)   | [link](https://drive.google.com/file/d/1BLL7sDE_Hg1rAW-mKO3YZzkGiTsO-wTv/view?usp=sharing)   | [link](https://drive.google.com/file/d/1hbf0XfSovBkvHPP6Ys65fwO2L7vf59l0/view?usp=sharing)   |
### The test results of our method and other methods mentioned in the paper.
| FE108 | FELT | VisEvent |
|----------|----------|----------|
|  [link](https://drive.google.com/file/d/1a1nyrJH-6SNpryxSEYzcZptgbWWO4pV0/view?usp=sharing)  |  [link](https://drive.google.com/file/d/1c98n2EJDDlRIratJRozhkzVENuX--OOf/view?usp=sharing)  | [link](https://drive.google.com/file/d/1Ctll5AfGtjtXnP6HLCtJD3h1Zx4hXcqP/view?usp=sharing)   |

