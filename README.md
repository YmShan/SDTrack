<div align="center">
  <img src="https://github.com/YmShan/SDTrack/blob/main/source/logo.png">
</div>

# SDTrack: A Baseline for Neuromorphic Tracking via Spiking Neural Networks

# We have established comprehensive evaluation benchmarks for both event camera-based and spike camera-based tracking methodologies.

For the **event camera** component, please refer to [SDTrack-Event](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Event/readme.md).

For the **spike camera** component, please refer to [SDTrack-Spike](https://github.com/YmShan/SDTrack/blob/main/SDTrack-Spike/readme.md).

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
