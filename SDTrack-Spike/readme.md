## Conda Environment
You can download the compressed package of the environment [here](https://drive.google.com/file/d/1bHu7CbM6TiSXNXnMbfj8W-eUNvO_4wyA/view?usp=sharing). Or manually install the environment using [create_SDTrack_env.sh](https://github.com/YmShan/SDTrack/blob/main/create_SDTrack_env.sh)

## Data Prepare
1. Download [Spike-FE108](https://pan.baidu.com/s/1PzHsg3zpU5NF8-vB1TVmaQ?pwd=7hsf), [Spike-VisEvent](https://pan.baidu.com/s/1YouDIMm0otCtI6p2oqAqJg?pwd=ixjx) and [Spike-COESOT](https://pan.baidu.com/s/15KDBaV3-z0lGyBFJlSgCgw?pwd=u7cv).
2. Place the three scripts in the following paths accordingly:
```
├── Spike-FE108
    ├── train
        ├── airplane
            ├── spike_thresh_2.5_decay_0.25
            ├── groundtruth_rect.txt
        ├── airplane222
        ├── ...
        ├── whale_mul111
    ├── test
        ├── airplane_mul222
            ├── spike_thresh_2.5_decay_0.25
            ├── groundtruth_rect.txt
        ├── bike222
        ├── ...
        ├── whale_mul222
├── Spike-VisEvent
    ├── train
        ├── 00143_tank_outdoor2
            ├── spike_thresh_2.5_decay_0.25
            ├── groundtruth.txt
        ├── 00145_tank_outdoor2
        ├── ...
        ├── video_0081
    ├── test
        ├── 00141_tank_outdoor2
            ├── spike_thresh_2.5_decay_0.25
            ├── groundtruth.txt
        ├── 00147_tank_outdoor2
        ├── ...
        ├── video_0079
├── Spike-COESOT
    ├── training_subset
        ├── dvSave-2021_09_01_06_59_10
            ├── spike_thresh_2.5_decay_0.25
            ├── groundtruth.txt
        ├── dvSave-2021_09_01_07_00_26
        ├── ...
        ├── dvSave-2022_03_21_16_27_10
    ├── testing_subset
        ├── dvSave-2021_07_30_11_04_12
            ├── spike_thresh_2.5_decay_0.25
            ├── groundtruth.txt
        ├── dvSave-2021_07_30_11_04_57
        ├── ...
        ├── dvSave-2022_09_24_16_05_56
```
All spike data are constructed using the [video2spike_3C.py](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Spike/video2spike_3C.py) script.
## Download the pre-trained weights from ImageNet-1K.
1. Download [SDTrack-Tiny](https://drive.google.com/file/d/1OcXHCnibEv9F40gw5VwGO90adtE6E0Ik/view?usp=sharing) and [SDTrack-Base](https://drive.google.com/file/d/1maJd0td46oxHACeBk2Vc90a__VyDAeWj/view?usp=sharing).
2. Create the directory SDTrack/**pretrained_models** and place the two downloaded weight files into this directory.

## Modify the settings required for training and testing.
1. The training path configuration file is located at `SDTrack/lib/train/admin/local.py`.
2. The testing path configuration file is located at `SDTrack/lib/test/evaluation/local.py`.

## Training
To mitigate training pressure, enhance generalization capability, and fully exploit the network's learning capacity, our constructed Spike-based Tracking Baseline employs simultaneous training across three datasets.
```
# Training the Tiny model (1 * RTX4090 = 1day)
bash train_tiny.sh
# Training the Base model
bash train_base.sh
```

## Test
Testing on all three datasets utilizes the same script
```
# Testing the Tiny model
bash test_tiny.sh
# Testing the Base model
bash test_base.sh
```
It is noteworthy that since we conduct training only once, the test results from all three datasets will be stored in the same folder. Therefore, we recommend downloading the test results locally after completing the evaluation on the first dataset, then deleting the test results of the first dataset before proceeding to evaluate the second dataset, and so forth.


## Evaluation
1. Download the MATLAB script for evaluation([FE108](https://drive.google.com/file/d/1sf2pSOAYAcsWbnxC2brsG_QnzvMP0rrJ/view?usp=sharing), [VisEvent](https://drive.google.com/file/d/1QgZEMbnJifpSFjnUJIVlL9D3_AeOZWYf/view?usp=sharing)) and [COESOT](https://drive.google.com/file/d/1LR_9PgqlsxrSKfIKpT84gmWUHF_LBrcC/view?usp=sharing)). The evaluation scripts for FELT and VisEvent were provided by [Xiao Wang](https://github.com/wangxiao5791509), while the evaluation script for FE108 was modified by us.
2. For the three datasets, before evaluation, the test results (including multiple .txt files) need to be copied to the `tracking_results` folder in the corresponding directory. Additionally, the `utils/config_tracker.m` file in the respective folder should be modified. Finally, run the corresponding MATLAB script to generate the evaluation results. It is important to note that before testing AUC, you need to set `ranking_type = AUC`, and before testing PR, you need to set `ranking_type = threshold`. 

## SDTrack Event-based Tracking Baseline
| Methods        | Param. (M) | Spiking Neuron | Timesteps (T × D) | Power (mJ) | FE108 AUC(%) | FE108 PR(%) | VisEvent AUC(%) | VisEvent PR(%) | COESOT AUC(%) | COESOT PR(%) |Weight|
|:----------------:|:------------:|:----------------:|:-------------------:|:------------:|:--------------:|:-------------:|:-------------:|:------------:|:-----------------:|:----------------:|:-:|
| SimTrack | 88.64 | -          | 1 × 1            |   93.84    |  78.3     | 52.1   |  62.1   | 44.8  |  69.8     |  50.3     |[Link](https://drive.google.com/file/d/1u8vMDMgxQidAQ2o_HXev0aAV0udB8e-a/view?usp=sharing)|
| OSTrack | 92.52 | -          | 1 × 1            |  98.90   | 70.7   |46.4  | 65.6   |47.8  |   76.4   |  56.7   |[Link](https://drive.google.com/file/d/1R4v-X29k-sXqhP1iUS2jnRBx26lUYxy4/view?usp=sharing)|
| STARK |28.23 | -          | 1 × 1            | 58.88   |  72.4    | 48.2 | 55.3  | 40.0| 63.0    | 50.1     |[Link](https://drive.google.com/file/d/1g04hgiaA07kfYmwGrr4OLycPj7e8uoQV/view?usp=sharing)|
| HIT| | -          | 1 × 1            |    |      |  |   | |     |      ||
| GRM | 99.83| -          | 1 × 1            | 142.14   |  75.0    | 49.9 | 67.2  |49.1 |  79.3   |  58.9    |[Link](https://drive.google.com/file/d/1GCl7fGkMMKoy9rINy-JiLYC9tp_oFREj/view?usp=sharing)|
| SeqTrack | | -          | 1 × 1            |    |      |  |   | |     |      ||
| HIPTrack | | -          | 1 × 1            |    |      |  |   | |     |      ||
| ARTrack | | -          | 1 × 1            |    |      |  |   | |     |      ||
| ODTrack | | -          | 1 × 1            |    |      |  |   | |     |      ||
| **SDTrack-Tiny** | 19.61 | I-LIF          | 1 × 4             |  3.75      |   71.7       |   47.1      |    59.1     |   42.3     |   67.2          |   50.3         |[Link](https://drive.google.com/file/d/13Vpan239XkEH03ZoPTyGYQ_JMPgKoRDv/view?usp=sharing)|

For SDTrack-Tiny's firing rate and energy consumption calculations, see [Energy_Consumption.py](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Spike/Energy_Consumption.py).

For SDTrack-Tiny: The downloaded weights should be placed in the SDTrack-Spike/output/checkpoints/train/SDTrack/SDTrack-tiny directory. Subsequently, testing can be executed directly.

## Get the inference results.


### Tracking Result (Put which into a MATLAB script for testing)
|Tracker|Spike-FE108|Spike-VisEvent|Spike-COESOT|
|:-:|:-:|:-:|:-:|
|SimTrack|||
|OSTrack|||
|STARK|||
|HIT|||
|GRM|||
|SeqTrack|||
|HIPTrack|||
|ARTrack|||
|ODTrack|||
|SDTrack-Tiny|[Link](https://drive.google.com/file/d/1mdA4pKtX4AOY4EzsRnlgO3dX-l1mUh8K/view?usp=sharing)|[Link](https://drive.google.com/file/d/1GVKxlTXcSRynsSlSVfu3mpOjCTXz_JVj/view?usp=sharing)|[Link](https://drive.google.com/file/d/1sB1ziRP7QEioZayfWuEn54VgYfE8MXRY/view?usp=sharing)|

