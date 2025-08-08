<div align="center">
  <img src="https://github.com/YmShan/SDTrack/blob/main/source/logo.png">
</div>

# SDTrack: A Baseline for Neuromorphic Tracking via Spiking Neural Networks

# We have established comprehensive evaluation benchmarks for both event camera-based and spike camera-based tracking methodologies.

For the **event camera** component, please refer to [访问 OpenAI 官网](https://openai.com).
For the **spike camera** component, please refer to [访问 OpenAI 官网](https://openai.com).

## Tracker Overview
<div align="center">
  <img src="https://github.com/YmShan/SDTrack/blob/main/source/main_pic.png">
</div>


## Conda Environment
You can download the compressed package of the environment [here](https://drive.google.com/file/d/1bHu7CbM6TiSXNXnMbfj8W-eUNvO_4wyA/view?usp=sharing). Or manually install the environment using `create_SDTrack_env.sh`.

## Data Prepare
1. The processing of the FELT dataset is relatively intricate; thus, we recommend utilizing only the FE108 and VisEvent datasets at this stage. Results on the COESOT dataset will be provided in our forthcoming updates.
2. Download [FE108](https://zhangjiqing.com/dataset/), [FELT](https://github.com/Event-AHU/FELT_SOT_Benchmark) and [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark).
3. Download the datasets processing scripts for the three datasets ([FE108](https://drive.google.com/file/d/1OXMXYbRsQIoxMujkJ-K3cxdfpRog5Ca7/view?usp=sharing), [FELT](https://drive.google.com/file/d/1SApVrzb90sP_D8wYFOpOMwsmCeOMMXhG/view?usp=sharing) and [VISEVENT](https://drive.google.com/file/d/17zm3HjA6iPLmY0chKRwMYEmxUD1IAosG/view?usp=sharing)).
4. Place the three scripts in the following paths accordingly:
```
├── FE108
    ├── train
        ├── airplane
            ├── events.aedat4
            ├── groundtruth_rect.txt
        ├── airplane222
        ├── ...
        ├── whale_mul111
    ├── test
        ├── airplane_mul222
            ├── events.aedat4
            ├── groundtruth_rect.txt
        ├── bike222
        ├── ...
        ├── whale_mul222
    ├── GTP_FE108.py
├── VisEvent
    ├── train
        ├── 00143_tank_outdoor2
            ├── 00143_tank_outdoor2.aedat4
            ├── groundtruth.txt
        ├── 00145_tank_outdoor2
        ├── ...
        ├── video_0081
    ├── test
        ├── 00141_tank_outdoor2
            ├── 00141_tank_outdoor2.aedat4
            ├── groundtruth.txt
        ├── 00147_tank_outdoor2
        ├── ...
        ├── video_0079
    ├── GTP_VisEvent.py
├── FELT
    ├── train
        ├── dvSave-2022_10_11_19_24_36
            ├── dvSave-2022_10_11_19_24_36.aedat4
            ├── groundtruth.txt
        ├── dvSave-2022_10_11_19_27_02
        ├── ...
        ├── dvSave-2022_10_31_10_56_34
    ├── test
        ├── dvSave-2022_10_11_19_43_03
            ├── dvSave-2022_10_11_19_43_03.aedat4
            ├── groundtruth.txt
        ├── dvSave-2022_10_11_19_51_27
        ├── ...
        ├── dvSave-2022_10_31_10_52_10
    ├── GTP_FELT.py
```

4.Run the three scripts：
```
python YOUR_FE108_PATH/GTP_FE108.py --trans_folder 0 --source_dir YOUR_FE108_ROOT_PATH --target_dir YOUR_FE108_ROOT_PATH --stack_name inter1_stack_3008 --s_train 0 --e_train 76 --s_test 0 --e_test 32 --stack_amount_1c2c 30 --stack_amount_3c 30 --decay_rate_3c 0.8
```
```
python YOUR_FELT_PATH/GTP_FELT.py --trans_folder 0 --source_dir YOUR_FELT_ROOT_PATH --target_dir YOUR_FELT_ROOT_PATH --stack_name inter1_stack_3008 --s_train 0 --e_train 76 --s_test 0 --e_test 32 --stack_amount_1c2c 30 --stack_amount_3c 30 --decay_rate_3c 0.8
```
```
python YOUR_VisEvent_PATH/GTP_VisEvent.py --trans_folder 0 --source_dir YOUR_VisEvent_ROOT_PATH --target_dir YOUR_VisEvent_ROOT_PATH --stack_name inter1_stack_3008 --s_train 0 --e_train 76 --s_test 0 --e_test 32 --stack_amount_1c2c 30 --stack_amount_3c 30 --decay_rate_3c 0.8
```

## Download the pre-trained weights from ImageNet-1K.
1. Download [SDTrack-Tiny](https://drive.google.com/file/d/1OcXHCnibEv9F40gw5VwGO90adtE6E0Ik/view?usp=sharing) and [SDTrack-Base](https://drive.google.com/file/d/1maJd0td46oxHACeBk2Vc90a__VyDAeWj/view?usp=sharing).
2. Create the directory SDTrack/**pretrained_models** and place the two downloaded weight files into this directory.

## Modify the settings required for training and testing.
1. The training path configuration file is located at `SDTrack/lib/train/admin/local.py`.
2. The testing path configuration file is located at `SDTrack/lib/test/evaluation/local.py`.

## Training
```
# FE108
bash train_tiny_fe108.sh
bash train_base_fe108.sh
# VisEvent
bash train_tiny_visevent.sh
bash train_base_visevent.sh
# FELT
bash train_tiny_felt.sh
bash train_base_felt.sh
```

## Test
```
# FE108
bash test_tiny_fe108.sh
bash test_base_fe108.sh
# VisEvent
bash test_tiny_visevent.sh
bash test_base_visevent.sh
# FELT
bash test_tiny_felt.sh
bash test_base_felt.sh
```

## Before Running SDTrack-Tiny/Base On FELT Dataset.
1. **Transform Configuration Adjustment** Modify the data augmentation settings in the '/SDTrack/lib/train/base_functions.py' path to:
```
transform_train = tfm.Transform(tfm.ToTensor(), 
                               tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
                                )
transform_val = tfm.Transform(tfm.ToTensor(),
                              tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
                              )
```
2. **ToTensor Class Modification** Revise the transform_image method in the ToTensor class located at '/SDTrack/lib/train/data/transforms.py' to:
```
def transform_image(self, image):
    # handle numpy array
    if image.ndim == 2:
        image = image[:, :, None]

    image = torch.from_numpy(image.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(image, torch.ByteTensor):
        return image.float().div(255)
    else:
        return image
```
3. **Testing Phase Modification** Alter the Preprocessor class in '/SDTrack/lib/test/tracker/data_utils.py' to:
```
class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)
```


## Evaluation
1. Download the MATLAB script for evaluation([FE108](https://drive.google.com/file/d/1sf2pSOAYAcsWbnxC2brsG_QnzvMP0rrJ/view?usp=sharing), [FELT](https://drive.google.com/file/d/1CqYK8q2mysR2FGZx9GJWY6lzbXSiUXxF/view?usp=sharing) and [VisEvent](https://drive.google.com/file/d/1QgZEMbnJifpSFjnUJIVlL9D3_AeOZWYf/view?usp=sharing)). The evaluation scripts for FELT and VisEvent were provided by [Xiao Wang](https://github.com/wangxiao5791509), while the evaluation script for FE108 was modified by us.
2. For the three datasets, before evaluation, the test results (including multiple .txt files) need to be copied to the `tracking_results` folder in the corresponding directory. Additionally, the `utils/config_tracker.m` file in the respective folder should be modified. Finally, run the corresponding MATLAB script to generate the evaluation results. It is important to note that before testing AUC, you need to set `ranking_type = AUC`, and before testing PR, you need to set `ranking_type = threshold`. For the FELT dataset, before moving the test results to the `tracking_results` folder, you first need to move the test results to the `processing_data` directory and run `processing_1.py` and `processing_2.py` to correct their format.


## Get the training and inference results.
### Weights
|  | FE108 | FELT | VisEvent |
|----------|----------|----------|----------|
| SDTrack-Tiny    |  [link](https://drive.google.com/file/d/1Hal0RcEgYKuqBiUFwPHa8f2bisboIp80/view?usp=sharing)  |  [link](https://drive.google.com/file/d/1GoGljfudnjSw7bvW53bpPy2jv2-IZstd/view?usp=sharing)  | [link](https://drive.google.com/file/d/1rbZT2DBMeKrWZ8ORwNDz9fBKoMqRGN-_/view?usp=sharing)   |
| SDTrack-Base    | [link](https://drive.google.com/file/d/1tnJme3hugllA8xAIODoARzKaOkQKh6jr/view?usp=sharing)   | [link](https://drive.google.com/file/d/18deLeGd2hWOtdU2C6YoxHrSTseIPfKyv/view?usp=sharing)   | [link](https://drive.google.com/file/d/1hbf0XfSovBkvHPP6Ys65fwO2L7vf59l0/view?usp=sharing)   |
### The test results of our method and other methods mentioned in the paper.
| FE108 | FELT | VisEvent |
|----------|----------|----------|
|  [link](https://drive.google.com/file/d/1Slse96Gu3m0RCpAa0vwZq1nC4iVxAoT9/view?usp=sharing)  |  [link](https://drive.google.com/file/d/1XIXwD7PWk-WUcliqi5DMJzJ4X-jassDt/view?usp=sharing)  | [link](https://drive.google.com/file/d/1By9Wh_L0d8gOxl12_b3T4XaKoOW0CXx1/view?usp=drive_link)   |

## Citation
If you use our tracker(SDTrack), Event Aggregation method(GTP), reported experimental results, or any other original content from this work, please cite our paper:
```
@article{shan2025sdtrack,
  title={Sdtrack: A baseline for event-based tracking via spiking neural networks},
  author={Shan, Yimeng and Ren, Zhenbang and Wu, Haodi and Wei, Wenjie and Zhu, Rui-Jie and Wang, Shuai and Zhang, Dehao and Xiao, Yichen and Zhang, Jieyuan and Shi, Kexin and others},
  journal={arXiv preprint arXiv:2503.08703},
  year={2025}
}
```

## Handling of errors caused by device-related issues.
### 1. 'local-rank' error.
Change the code in 'SDTrack/lib/train/run_training.py'
```python
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
```
to
```python
parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
```

### 2. 'torch.six' Error.
Change the code in 'SDTrack/lib/train/data/loader.py'
```python
from torch._six import string_classes
```
to
```python
string_classes = str
```

### 3. Frequent warning reports.
```
/xxx/SDTrack/lib/train/../../lib/train/data/loader.py:88: UserWarning: An output with one or more elements was resized since it had shape [3145728], which does not match the required output shape [1, 16, 3, 256, 256]. This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). (Triggered internally at /opt/conda/conda-bld/pytorch_1720538622298/work/aten/src/ATen/native/Resize.cpp:28.)
```
Add the code in the top of '/SDTrack/lib/train/run_training.py'
```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
```
