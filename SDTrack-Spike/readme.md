## Conda Environment
You can download the compressed package of the environment [here](https://drive.google.com/file/d/1bHu7CbM6TiSXNXnMbfj8W-eUNvO_4wyA/view?usp=sharing). Or manually install the environment using [create_SDTrack_env.sh](https://github.com/YmShan/SDTrack/blob/main/create_SDTrack_env.sh)

## Data Prepare
1. Download [Spike-FE108](https://pan.baidu.com/s/1PzHsg3zpU5NF8-vB1TVmaQ?pwd=7hsf), [Spike-VisEvent](https://pan.baidu.com/s/1YouDIMm0otCtI6p2oqAqJg?pwd=ixjx) and [Spike-COESOT](https://pan.baidu.com/s/15KDBaV3-z0lGyBFJlSgCgw?pwd=u7cv).
2. The correct directory structure for the three datasets after extraction is as follows:
```
├── Spike-FE108
    ├── train
        ├── airplane
            ├── spike_thresh_2.5_decay_0.25
                ├── 0000.png
                ├── 0001.png
                ├── ...
                ├── 1113.png
            ├── groundtruth_rect.txt
        ├── airplane222
        ├── ...
        ├── whale_mul111
    ├── test
        ├── airplane_mul222
            ├── spike_thresh_2.5_decay_0.25
                ├── 0000.png
                ├── 0001.png
                ├── ...
                ├── 2050.png
            ├── groundtruth_rect.txt
        ├── bike222
        ├── ...
        ├── whale_mul222
├── Spike-VisEvent
    ├── train
        ├── 00143_tank_outdoor2
            ├── spike_thresh_2.5_decay_0.25
                ├── 0000.png
                ├── 0001.png
                ├── ...
                ├── 0103.png
            ├── groundtruth.txt
        ├── 00145_tank_outdoor2
        ├── ...
        ├── video_0081
    ├── test
        ├── 00141_tank_outdoor2
            ├── spike_thresh_2.5_decay_0.25
                ├── 0000.png
                ├── 0001.png
                ├── ...
                ├── 0104.png
            ├── groundtruth.txt
        ├── 00147_tank_outdoor2
        ├── ...
        ├── video_0079
├── Spike-COESOT
    ├── training_subset
        ├── dvSave-2021_09_01_06_59_10
            ├── spike_thresh_2.5_decay_0.25
                ├── 0000.png
                ├── 0001.png
                ├── ...
                ├── 0037.png
            ├── groundtruth.txt
        ├── dvSave-2021_09_01_07_00_26
        ├── ...
        ├── dvSave-2022_03_21_16_27_10
    ├── testing_subset
        ├── dvSave-2021_07_30_11_04_12
            ├── spike_thresh_2.5_decay_0.25
                ├── 0000.png
                ├── 0001.png
                ├── ...
                ├── 0492.png
            ├── groundtruth.txt
        ├── dvSave-2021_07_30_11_04_57
        ├── ...
        ├── dvSave-2022_09_24_16_05_56
```
All spike data are constructed using the [video2spike_3C.py](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Spike/video2spike_3C.py) script.

Please kindly note the storage space requirements. The compressed versions of the three datasets are 20.98GB, 63.12GB, and 111.56GB, respectively, with their uncompressed sizes being 26GB, 89GB, and 139GB, respectively.

## Download the pre-trained weights from ImageNet-1K.
1. Download [SDTrack-Tiny](https://drive.google.com/file/d/1OcXHCnibEv9F40gw5VwGO90adtE6E0Ik/view?usp=sharing) and [SDTrack-Base](https://drive.google.com/file/d/1maJd0td46oxHACeBk2Vc90a__VyDAeWj/view?usp=sharing).
2. Create the directory SDTrack/**pretrained_models** and place the two downloaded weight files into this directory.

## Modify the settings required for training and testing.
1. The training path configuration file is located at `SDTrack/lib/train/admin/local.py`.
2. The testing path configuration file is located at `SDTrack/lib/test/evaluation/local.py`.

The entire modification process consists of two steps. The first step is to replace `/data1/users/xxx/SDTrack-Spike` with your file path. The second step is to align the dataset paths with your dataset storage locations:
```
self.eotb_dir_train = '/data1/dataset/Spike-FE108/train'
self.visevent_train = '/data1/dataset/Spike-VisEvent/train/'
self.coesot_train = '/data/dataset/Spike-COESOT/training_subset/'

settings.eotb_path = '/data1/dataset/FE108/test'
settings.visevent_path = '/data1/dataset/VisEvent/test/'
settings.coesot_path = '/data/dataset/COESOT_dataset/testing_subset/'
```

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
1. Download the MATLAB script for evaluation([FE108](https://drive.google.com/file/d/1sf2pSOAYAcsWbnxC2brsG_QnzvMP0rrJ/view?usp=sharing), [VisEvent](https://drive.google.com/file/d/1QgZEMbnJifpSFjnUJIVlL9D3_AeOZWYf/view?usp=sharing)) and [COESOT](https://drive.google.com/file/d/1LR_9PgqlsxrSKfIKpT84gmWUHF_LBrcC/view?usp=sharing)). The evaluation scripts for Spike-VisEvent were provided by [Xiao Wang](https://github.com/wangxiao5791509), while the evaluation script for Spike-FE108 and Spike-COESOT was modified by us.
2. For the three datasets, before evaluation, the test results (including multiple .txt files) need to be copied to the `tracking_results` folder in the corresponding directory. Additionally, the `utils/config_tracker.m` file in the respective folder should be modified. Finally, run the corresponding MATLAB script to generate the evaluation results. It is important to note that before testing AUC, you need to set `ranking_type = AUC`, and before testing PR, you need to set `ranking_type = threshold`. 

If MATLAB configuration proves challenging, an alternative testing approach can be utilized `python tracking/analysis_results.py`. However, please note that the test results may exhibit some deviation from our reported results due to differences in integration calculation methods.


## SDTrack Event-based Tracking Baseline
| Methods        | Param. (M)  | Timesteps (T × D) | Power (mJ) | FE108 AUC(%) | FE108 PR(%) | VisEvent AUC(%) | VisEvent PR(%) | COESOT AUC(%) | COESOT PR(%) |Code|Weight|
|:----------------:|:----------------:|:-------------------:|:------------:|:--------------:|:-------------:|:-------------:|:------------:|:-----------------:|:----------------:|:-:|:-:|
| SimTrack | 88.64 | 1 × 1            |   93.84    |  78.3     | 52.1   |  62.1   | 44.8  |  69.8     |  50.3  |  [Link](https://drive.google.com/file/d/1-YU8QBLH48BkUvgUmXaQV41_tF-JWInZ/view?usp=sharing) |[Link](https://drive.google.com/file/d/1u8vMDMgxQidAQ2o_HXev0aAV0udB8e-a/view?usp=sharing)|
| OSTrack | 92.52  | 1 × 1            |  98.90   | 70.7   |46.4  | 65.6   |47.8  |   76.4   |  56.7  | [Link](https://drive.google.com/file/d/1BpAM5EGJXEckGp5ZIaeLv1tQZF5jZusj/view?usp=sharing)|[Link](https://drive.google.com/file/d/1R4v-X29k-sXqhP1iUS2jnRBx26lUYxy4/view?usp=sharing)|
| STARK |28.23 | 1 × 1            | 58.88   |  72.4    | 48.2 | 55.3  | 40.0| 63.0    | 50.1    | [Link](https://drive.google.com/file/d/1lLGLLYF5Not_Ro3SkhF0Bvj3AMIE1CeR/view?usp=sharing)|[Link](https://drive.google.com/drive/folders/1DN2nflGKXM7Ho_HZ7FFiWg2t1MZTmW3Y?usp=sharing)|
| HIT| 42.22| 1 × 1            |  19.78  |   62.1   | 39.2 |  48.1 | 34.7| 51.3    | 42.3| [Link](https://drive.google.com/file/d/1NCcBpTd2d0TY5krnSCmTC_PyXMGG2bhd/view?usp=sharing)    |[Link](https://drive.google.com/file/d/17usRyETw4HDX0OfiJ4LhqC-zdBi8YEKF/view?usp=sharing)|
| GRM | 99.83| 1 × 1            | 142.14   |  75.0    | 49.9 | 67.2  |49.1 |  79.3   |  58.9  | [Link](https://drive.google.com/file/d/1QGCVi-WMGHJTi0taVO3_nUmxO2jcfFvX/view?usp=sharing) |[Link](https://drive.google.com/file/d/1GCl7fGkMMKoy9rINy-JiLYC9tp_oFREj/view?usp=sharing)|
| SeqTrack | 90.60 | 1 × 1            |   302.68 |  72.8    | 47.3 | 65.6  | 47.3| 76.5    | 56.8     |[Link](https://drive.google.com/file/d/1AfGlckEQOuqUCgenjbvcXKuDjRpFE7kx/view?usp=sharing)|[Link](https://drive.google.com/file/d/1oc-KtKGA4_3TLQks8iYbiSgAJmsOmxol/view?usp=sharing)|
| HIPTrack | 120.41 | 1 × 1            | 307.74   |  69.6    | 45.3 |67.4   |49.6 | 82.5    |  62.5    |[Link](https://drive.google.com/file/d/1lK2Ahwx29PJXfF-b4w0qhwNKM3yXCDY_/view?usp=sharing)|[Link](https://drive.google.com/file/d/1Di9p-iWzQs-k0WJKZjJghnHeDjRdq1Wk/view?usp=sharing)|
| ODTrack |  92.83 | 1 × 1            |  335.80  |  78.7    | 52.2 |68.5   | 50.4| 75.5    | 56.9     |[Link](https://drive.google.com/file/d/1gZ33PYMYE3AFQw2YAO2zG5sOgz3FUrzs/view?usp=sharing)|[Link](https://drive.google.com/file/d/1TeY8IsxOkR5CL3evxhiXAYGHU08YEmsr/view?usp=sharing)|
| **SDTrack-Tiny** | 19.61 | 1 × 4             |  3.75      |   71.7       |   47.1      |    59.1     |   42.3     |   67.2          |   50.3       |[Link](https://github.com/YmShan/SDTrack/tree/main/SDTrack-Spike)  |[Link](https://drive.google.com/file/d/13Vpan239XkEH03ZoPTyGYQ_JMPgKoRDv/view?usp=sharing)|

For SDTrack-Tiny's firing rate and energy consumption calculations, see [Energy_Consumption.py](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Spike/Energy_Consumption.py).

For SDTrack-Tiny: The downloaded weights should be placed in the SDTrack-Spike/output/checkpoints/train/SDTrack/SDTrack-tiny directory. Subsequently, testing can be executed directly.

## Get the inference results.


### Tracking Result (Put which into a MATLAB script for testing)
|Tracker|Spike-FE108|Spike-VisEvent|Spike-COESOT|
|:-:|:-:|:-:|:-:|
|SimTrack|[Link](https://drive.google.com/file/d/1GX-X6Onpz8o5A4b1rlR5R_Uzde5ZkWv0/view?usp=sharing)|[Link](https://drive.google.com/file/d/1owWPyp4yb-Zc2KrThfa3Ca4BrtIIZdO5/view?usp=sharing)|[Link](https://drive.google.com/file/d/1TjidTo-LooXy4CgZ280C7L84XuEmgDFh/view?usp=sharing)|
|OSTrack|[Link](https://drive.google.com/file/d/1ykHG5X3sFGTmUVg61gb8cfjuF-3EsaRB/view?usp=sharing)|[Link](https://drive.google.com/file/d/1bDhTv3MqztnyXuqShmjUHeqQ5IihSUWe/view?usp=sharing)|[Link](https://drive.google.com/file/d/1kSYR26NDWzr8tSpPb2X3rTUgbfwidxf3/view?usp=sharing)|
|STARK|[Link](https://drive.google.com/file/d/1xloKyvUIiKlgYTpNPPbM_ShTFlDt915c/view?usp=sharing)|[Link](https://drive.google.com/file/d/1kFQLpIdS4AmWeGLaA32ua97W9feV2Dbh/view?usp=sharing)|[Link](https://drive.google.com/file/d/1G60T9XnEDLBqpS78FJpnZne89nWHOW4i/view?usp=sharing)|
|HIT|[Link](https://drive.google.com/file/d/1MBmF26qDR2Y0y9d1KDZUmUJl9cr8Dyv1/view?usp=sharing)|[Link](https://drive.google.com/file/d/1OzJJTd6B-IN8D5nA6TUvNtZZY4gVRhwm/view?usp=sharing)|[Link](https://drive.google.com/file/d/1pTUQO_ujS13DWQFw5B-9S5DS4BDzyztv/view?usp=sharing)|
|GRM|[Link](https://drive.google.com/file/d/1rVf6iabk0PL5muRiw9_Uz0pRStj6p9an/view?usp=sharing)|[Link](https://drive.google.com/file/d/1IIwzukinawwLMrNgj3SKsEK142tUExT3/view?usp=sharing)|[Link](https://drive.google.com/file/d/1pvspZLRFtj3TF5U8W6VeaAbDQ1sOf4TZ/view?usp=sharing)|
|SeqTrack|[Link](https://drive.google.com/file/d/1729k-ywulEe2qwM0dVi10ItrTjJk7tU1/view?usp=sharing)|[Link](https://drive.google.com/file/d/1N-aZDzVMAFNl1q05NQZ57ekM719LcBah/view?usp=sharing)|[Link](https://drive.google.com/file/d/1dxf0tjdjWajYbBpUHHHn78EE_r5JsnJ_/view?usp=sharing)|
|HIPTrack|[Link](https://drive.google.com/file/d/1RWHx5HQycYwoMhB0ecHDEoZwfstkoNhP/view?usp=sharing)|[Link](https://drive.google.com/file/d/1C6ig8EYjgmqbvIjLbxi4SVa5lPsg22vw/view?usp=sharing)|[Link](https://drive.google.com/file/d/1kC_aFCHLvvvsFmRW1ABI-jn9NlkURClf/view?usp=sharing)|
|ODTrack|[Link](https://drive.google.com/file/d/14tAZY0Fw0vxAg9NbSkkVW8n-GjVby1up/view?usp=sharing)|[Link](https://drive.google.com/file/d/1uA7fdsx7eGemCQhlIyIiBHhmA_84-K75/view?usp=sharing)|[Link](https://drive.google.com/file/d/1ENUu10CEH0epspdnbFZge1hMMyts7_U_/view?usp=sharing)|
|SDTrack-Tiny|[Link](https://drive.google.com/file/d/1mdA4pKtX4AOY4EzsRnlgO3dX-l1mUh8K/view?usp=sharing)|[Link](https://drive.google.com/file/d/1GVKxlTXcSRynsSlSVfu3mpOjCTXz_JVj/view?usp=sharing)|[Link](https://drive.google.com/file/d/1sB1ziRP7QEioZayfWuEn54VgYfE8MXRY/view?usp=sharing)|

