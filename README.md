<div align="center">
  <img src="https://github.com/YmShan/SDTrack/blob/main/source/logo.png">
</div>

# SDTrack: A Baseline for Neuromorphic Tracking via Spiking Neural Networks

## We establish comprehensive benchmarks for event-based camera and spiking-camera tracking.

For the **event-based camera** component, please refer to [SDTrack-Event](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Event/readme.md).

For the **spike camera** component, please refer to [SDTrack-Spike](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Spike/readme.md).


## News
***2025-03-09*** 

SDTrack preprint is now available on [arXiv](https://arxiv.org/abs/2503.08703).

***2025-03-15*** 

SDTrack code is now available.

***2025-08-14*** 

:trophy: :trophy: :trophy: We extended SDTrack to spiking cameras, introduced three datasets generated using spiking-camera simulators, and provided the first tracking baseline for spiking cameras. This work received the **Best Dataset & Benchmark Award** (sole winner) in the **IJCAI 2025 SpikeCV–Wuji Challenge (Dataset & Benchmark Track)**, along with a **$1,500 prize**. We thank the SpikeCV organizing committee.

***2025-09-19*** 

At **NeurIPS 2026**, **4** accepted papers ([NeurIPS-1](https://arxiv.org/pdf/2509.24266), [NeurIPS-2](https://openreview.net/forum?id=nG45z7lJ7D), [NeurIPS-3](https://arxiv.org/pdf/2510.21403), [NeurIPS-4](https://arxiv.org/pdf/2505.20834)) built upon our model or adopted our reported baseline. Congratulations to the authors, and thank you for your interest!
  
***2025-12-19*** 

A neuromorphic tracking system built upon the DAVIS346 camera, the SDTrack pipeline, and Lynxi chips was showcased at the **SLAI (Shenzhen Loop Area Institute) – LIMA (Language, Intelligence and Machines) Center Open Day**. [Link](https://mp.weixin.qq.com/s/nXbek5t8Wg3o_vyoZt1iNQ)

***2025-12-28***

The same neuromorphic tracking system was showcased at **the 5th Brain Science Frontier and Industry Conference and 2025 Shenzhen Brain-Computer Interface and Human-Machine Interaction Technology Expo**.


## Tracker Overview
<div align="center">
  <img src="https://github.com/YmShan/SDTrack/blob/main/source/main_pic.png">
</div>

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

### 4. Error with newer PyTorch versions: `weights_only` parameter in `torch.load`
```python
[rank0]:        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
[rank0]:        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
[rank0]:        WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace was not an allowed global by default. Please use `torch.serialization.add_safe_globals([argparse.Namespace])` or the `torch.serialization.safe_globals([argparse.Namespace])` context manager to allowlist this global if you trust this class/function.
```
Solution: Add the parameter `weights_only=False` to all `torch.load` functions.
