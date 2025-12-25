import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import string
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset  # 导入基本的视频数据集类
from ..admin import env_settings  # 导入环境设置
from ..data import opencv_loader  # 导入 OpenCV 加载器

# 定义不同类别的字典
cls = {'animal': ['dove', 'bear', 'elephant', 'cow', 'giraffe', 'dog', 'turtle', 'whale'],
       'vehicle': ['toy_car', 'airplane', 'fighter', 'truck', 'ship', 'tank', 'suv', 'bike'],
       'object': ['ball', 'star', 'cup', 'box', 'bottle', 'tower']}

# 定义visevent类继承BaseVideoDataset
class visevent(BaseVideoDataset):
    def __init__(self, root=None, image_loader=opencv_loader, split=None):
        """
        初始化函数
        args:
            root - 数据集的根路径
            image_loader - 用来读取图像的函数，默认使用 opencv 的 imread 函数
            split - 数据集的划分，可能为 'train' 或 'val'
        """
        root = env_settings().visevent_dir if root is None else root  # 如果没有提供根路径，则使用环境设置中的默认路径
        super().__init__('visevent', root, image_loader)  # 调用父类的初始化函数

        self.sequence_list = self._get_sequence_list()  # 获取序列列表，即所有文件夹的名称

        # 根据数据集划分读取训练集或验证集的文件路径
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'visevent_train_split_skx.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'visevent_val_split_skx.txt')
            else:
                raise ValueError('Unknown split name.')  # 如果split不为train或val，抛出错误
            with open(file_path) as f:
                seq_names = [line.strip() for line in f.readlines()]  # 从文件中读取序列名称
        else:
            seq_names = self.sequence_list  # 如果没有split参数，使用完整的数据集序列

        self.sequence_list = [i for i in seq_names]  # 更新sequence_list
        self.sequence_meta_info = self._load_meta_info()  # 加载每个序列的元信息
        self.seq_per_class = self._build_seq_per_class()  # 构建每个类别对应的序列字典
        self.class_list = list(self.seq_per_class.keys())  # 获取所有类别
        self.class_list.sort()  # 对类别列表进行排序

    def _build_seq_per_class(self):
        seq_per_class = {}

        # 遍历每个序列，按类别将序列划分到不同类别下
        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']  # 获取序列的对象类别名称
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class  # 返回每个类别对应的序列列表

    def _get_sequence_list(self):
        # 获取根目录下的所有序列文件夹
        seq_list = os.listdir(self.root)
        return seq_list  # 返回序列列表

    def _load_meta_info(self):
        # 加载每个序列的元信息
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info  # 返回序列元信息字典

    def get_num_classes(self):
        return len(self.class_list)  # 返回类别数量

    def get_name(self):
        return 'visevent'  # 返回数据集名称

    def get_class_list(self):
        # 返回类别列表
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def get_num_sequences(self):
        return len(self.sequence_list)  # 返回序列数量

    def _read_meta(self, seq_path):
        # 读取序列的元信息
        obj_class = self._get_class(seq_path)  # 获取对象类别
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta  # 返回对象元信息

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]  # 返回某个类别中的序列

    def _get_frame_path(self, seq_path, frame_id):
        # 根据采样频率确定帧号
        if '20fps' in seq_path:
            beishu = 12
        elif '15fps' in seq_path:
            beishu = 16
        elif '10fps' in seq_path:
            beishu = 24
        else:  # 默认灰度图像的采样频率为40Hz
            beishu = 6
        gray_id = frame_id // beishu + 1
        # 构建图像路径
        img_path = os.path.join(seq_path, 'img', '{:04}.jpg'.format(gray_id))
        return img_path

    def _get_frame(self, seq_path, frame_id):
        # 根据帧ID获取图像
        img_path = self._get_frame_path(seq_path, frame_id)
        return self.image_loader(img_path)  # 加载图像

    def _get_event(self, seq_path, frame_id, cfg=None):
        if cfg.MODEL.T == 1:
            event1 = os.path.join(seq_path, 'inter1_stack_3008', '{:04}_1.png'.format(frame_id))
            return self.image_loader(event1)  # 返回三个事件图像
        elif cfg.MODEL.T == 2:
            event1 = os.path.join(seq_path, 'inter2_stack_3008', '{:04}_1.png'.format(frame_id))
            event2 = os.path.join(seq_path, 'inter2_stack_3008', '{:04}_2.png'.format(frame_id))
            return self.image_loader(event1), self.image_loader(event2)
        elif cfg.MODEL.T == 4:
            event1 = os.path.join(seq_path, 'inter4_stack_3008', '{:04}_1.png'.format(frame_id))
            event2 = os.path.join(seq_path, 'inter4_stack_3008', '{:04}_2.png'.format(frame_id))
            event3 = os.path.join(seq_path, 'inter4_stack_3008', '{:04}_3.png'.format(frame_id))
            event4 = os.path.join(seq_path, 'inter4_stack_3008', '{:04}_4.png'.format(frame_id))
            return self.image_loader(event1), self.image_loader(event2), self.image_loader(event3), self.image_loader(event4)

    def _get_frames(self, seq_id):
        # 获取给定序列ID的帧图像
        path = self.coco_set.loadImgs([self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img  # 返回图像

    def _get_sequence_path(self, seq_id):
        # 获取序列路径
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_class_name(self, seq_id):
        # 获取序列的类别名称
        cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def _get_class(self, seq_path):
        # 获取序列的类别名称
        raw_class = seq_path.split('/')[-1].rstrip(string.digits).split('_')[0]
        return raw_class  # 返回类别名称

    def _read_bb_anno(self, seq_path):
        # 读取边界框注释文件
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)  # 返回边界框数据

    def get_sequence_info(self, seq_id):
        # 获取序列信息
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)  # 检查有效的边界框
        visible = valid.clone().byte()  # 可见性信息

        return {'bbox': bbox, 'valid': valid, 'visible': visible}  # 返回序列信息

    def get_frames(self, seq_id=None, frame_ids=None, anno=None, cfg=None):
        # 获取多帧图像和事件
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]  # 获取对象元信息

        # frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]  # 获取帧图像
        event_list = [self._get_event(seq_path, f_id, cfg) for f_id in frame_ids]  # 获取事件数据

        if anno is None:
            anno = self.get_sequence_info(seq_id)  # 获取注释信息

        anno_frames = {}
        # 为每一帧提取注释
        # {'bbox': [tensor([237.9454, 103.2980,  12.1810,  12.8113])], 'valid': [tensor(True)], 'visible': [tensor(1, dtype=torch.uint8)]}
        for key, value in anno.items():     
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]  # 克隆每帧注释

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(event_list[0][0].shape)           # [260, 346, 2]
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        return event_list, anno_frames, obj_meta  # 返回事件列表、注释和对象元信息          [[[260, 346, 2],[],[]]]
