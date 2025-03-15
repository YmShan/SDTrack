import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class FELTDataset(BaseDataset):
    """ OTB-2015 dataset

    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf

    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.felt_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']
        frames = []
        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        for frame_num in range(start_frame + init_omit, end_frame + 1):
            seq_path = '{base_path}/{sequence_path}/'.format(base_path=self.base_path,sequence_path=sequence_path)
            # frames.append(os.path.join(seq_path, 'frame{:04}.bmp'.format(frame_num)))
            frames.append(os.path.join(seq_path, '{:04}_1.png'.format(frame_num)))

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'FELT', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [{'anno_path': 'dvSave-2022_10_11_23_21_19/groundtruth.txt', 'endFrame': 1500, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_23_21_19', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_23_21_19/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_21_54_56/groundtruth.txt', 'endFrame': 2101, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_21_54_56', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_21_54_56/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_56_33/groundtruth.txt', 'endFrame': 2249, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_56_33', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_56_33/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_15_13_07/groundtruth.txt', 'endFrame': 1960, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_15_13_07', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_15_13_07/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_30_20_55_03/groundtruth.txt', 'endFrame': 1846, 'ext': 'jpg', 'name': 'dvSave-2022_10_30_20_55_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_30_20_55_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_21_55_12/groundtruth.txt', 'endFrame': 1888, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_21_55_12', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_21_55_12/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_35_44/groundtruth.txt', 'endFrame': 2425, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_35_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_35_44/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_19_49_53/groundtruth.txt', 'endFrame': 2361, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_19_49_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_19_49_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_18_21_55/groundtruth.txt', 'endFrame': 2097, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_18_21_55', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_18_21_55/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_20_04_18/groundtruth.txt', 'endFrame': 2192, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_20_04_18', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_20_04_18/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_17_12_23/groundtruth.txt', 'endFrame': 2371, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_17_12_23', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_17_12_23/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_19_36_03/groundtruth.txt', 'endFrame': 3097, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_19_36_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_19_36_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_20_09_31/groundtruth.txt', 'endFrame': 2474, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_20_09_31', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_20_09_31/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_18_24_31/groundtruth.txt', 'endFrame': 2132, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_18_24_31', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_18_24_31/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_00_10_09/groundtruth.txt', 'endFrame': 2121, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_00_10_09', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_00_10_09/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_21_26_59/groundtruth.txt', 'endFrame': 3442, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_21_26_59', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_21_26_59/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_23_12_14/groundtruth.txt', 'endFrame': 1948, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_23_12_14', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_23_12_14/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_18_23_33_48/groundtruth.txt', 'endFrame': 2039, 'ext': 'jpg', 'name': 'dvSave-2022_10_18_23_33_48', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_18_23_33_48/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_07_00/groundtruth.txt', 'endFrame': 277, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_07_00', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_07_00/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_23_06/groundtruth.txt', 'endFrame': 1825, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_23_06', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_23_06/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_21_57_32/groundtruth.txt', 'endFrame': 1868, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_21_57_32', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_21_57_32/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_17_14_50/groundtruth.txt', 'endFrame': 2109, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_17_14_50', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_17_14_50/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_38_12/groundtruth.txt', 'endFrame': 2578, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_38_12', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_38_12/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_19_39_09/groundtruth.txt', 'endFrame': 3263, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_19_39_09', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_19_39_09/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_58_45/groundtruth.txt', 'endFrame': 3060, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_58_45', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_58_45/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_20_08_52/groundtruth.txt', 'endFrame': 1839, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_20_08_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_20_08_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_23_25_00/groundtruth.txt', 'endFrame': 920, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_23_25_00', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_23_25_00/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_14_10_25_18/groundtruth.txt', 'endFrame': 1858, 'ext': 'jpg', 'name': 'dvSave-2022_10_14_10_25_18', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_14_10_25_18/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_15_49_50/groundtruth.txt', 'endFrame': 2082, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_15_49_50', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_15_49_50/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_19_53_02/groundtruth.txt', 'endFrame': 1963, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_19_53_02', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_19_53_02/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_15_15_10/groundtruth.txt', 'endFrame': 1800, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_15_15_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_15_15_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_18_24_51/groundtruth.txt', 'endFrame': 2003, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_18_24_51', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_18_24_51/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_29_22/groundtruth.txt', 'endFrame': 1555, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_29_22', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_29_22/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_30_21_10_12/groundtruth.txt', 'endFrame': 1826, 'ext': 'jpg', 'name': 'dvSave-2022_10_30_21_10_12', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_30_21_10_12/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_00_36_45/groundtruth.txt', 'endFrame': 1893, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_00_36_45', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_00_36_45/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_20_15_27/groundtruth.txt', 'endFrame': 1876, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_20_15_27', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_20_15_27/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_13_46_06/groundtruth.txt', 'endFrame': 2179, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_13_46_06', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_13_46_06/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_18_59_36/groundtruth.txt', 'endFrame': 2286, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_18_59_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_18_59_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_24_54/groundtruth.txt', 'endFrame': 1843, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_24_54', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_24_54/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_21_13_52/groundtruth.txt', 'endFrame': 2276, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_21_13_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_21_13_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_23_52_58/groundtruth.txt', 'endFrame': 1997, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_23_52_58', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_23_52_58/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_41_24/groundtruth.txt', 'endFrame': 2879, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_41_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_41_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_21_01_48/groundtruth.txt', 'endFrame': 2243, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_21_01_48', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_21_01_48/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_15_17_44/groundtruth.txt', 'endFrame': 1896, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_15_17_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_15_17_44/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_16_38_09/groundtruth.txt', 'endFrame': 1624, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_16_38_09', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_16_38_09/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_15_53_55/groundtruth.txt', 'endFrame': 1846, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_15_53_55', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_15_53_55/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_20_16_44/groundtruth.txt', 'endFrame': 2116, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_20_16_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_20_16_44/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_14_10_28_24/groundtruth.txt', 'endFrame': 2281, 'ext': 'jpg', 'name': 'dvSave-2022_10_14_10_28_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_14_10_28_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_19_34_48/groundtruth.txt', 'endFrame': 3125, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_19_34_48', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_19_34_48/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_20_10_53/groundtruth.txt', 'endFrame': 2497, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_20_10_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_20_10_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_18_45_24/groundtruth.txt', 'endFrame': 1989, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_18_45_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_18_45_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_20_30_15/groundtruth.txt', 'endFrame': 2803, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_20_30_15', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_20_30_15/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_17_16_58/groundtruth.txt', 'endFrame': 2082, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_17_16_58', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_17_16_58/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_21_22_16_32/groundtruth.txt', 'endFrame': 2153, 'ext': 'jpg', 'name': 'dvSave-2022_10_21_22_16_32', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_21_22_16_32/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_16_19_54/groundtruth.txt', 'endFrame': 2921, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_16_19_54', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_16_19_54/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_31_10_01_49/groundtruth.txt', 'endFrame': 1922, 'ext': 'jpg', 'name': 'dvSave-2022_10_31_10_01_49', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_31_10_01_49/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_21_20_26/groundtruth.txt', 'endFrame': 1990, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_21_20_26', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_21_20_26/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_00_39_02/groundtruth.txt', 'endFrame': 1970, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_00_39_02', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_00_39_02/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_19_34_01/groundtruth.txt', 'endFrame': 2238, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_19_34_01', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_19_34_01/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_19_38_36/groundtruth.txt', 'endFrame': 1876, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_19_38_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_19_38_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_15_56_06/groundtruth.txt', 'endFrame': 1921, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_15_56_06', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_15_56_06/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_13_48_27/groundtruth.txt', 'endFrame': 2102, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_13_48_27', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_13_48_27/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_21_17_01/groundtruth.txt', 'endFrame': 2049, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_21_17_01', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_21_17_01/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_20_17_36/groundtruth.txt', 'endFrame': 1914, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_20_17_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_20_17_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_33_35/groundtruth.txt', 'endFrame': 2828, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_33_35', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_33_35/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_50_45/groundtruth.txt', 'endFrame': 3625, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_50_45', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_50_45/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_43_35/groundtruth.txt', 'endFrame': 1980, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_43_35', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_43_35/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_20_14_09/groundtruth.txt', 'endFrame': 2104, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_20_14_09', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_20_14_09/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_18_47_34/groundtruth.txt', 'endFrame': 1835, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_18_47_34', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_18_47_34/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_14_11_01_24/groundtruth.txt', 'endFrame': 2790, 'ext': 'jpg', 'name': 'dvSave-2022_10_14_11_01_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_14_11_01_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_20_19_03/groundtruth.txt', 'endFrame': 2287, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_20_19_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_20_19_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_17_38_53/groundtruth.txt', 'endFrame': 2084, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_17_38_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_17_38_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_20_33_03/groundtruth.txt', 'endFrame': 2761, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_20_33_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_20_33_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_20_38_08/groundtruth.txt', 'endFrame': 2031, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_20_38_08', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_20_38_08/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_19_36_25/groundtruth.txt', 'endFrame': 2163, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_19_36_25', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_19_36_25/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_00_44_20/groundtruth.txt', 'endFrame': 1919, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_00_44_20', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_00_44_20/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_45_37/groundtruth.txt', 'endFrame': 2020, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_45_37', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_45_37/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_56_15/groundtruth.txt', 'endFrame': 2125, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_56_15', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_56_15/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_31_10_06_10/groundtruth.txt', 'endFrame': 1451, 'ext': 'jpg', 'name': 'dvSave-2022_10_31_10_06_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_31_10_06_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_13_50_35/groundtruth.txt', 'endFrame': 2372, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_13_50_35', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_13_50_35/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_21_22_18_47/groundtruth.txt', 'endFrame': 2148, 'ext': 'jpg', 'name': 'dvSave-2022_10_21_22_18_47', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_21_22_18_47/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_16_23_13/groundtruth.txt', 'endFrame': 2609, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_16_23_13', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_16_23_13/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_17_18_16/groundtruth.txt', 'endFrame': 3310, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_17_18_16', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_17_18_16/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_19_41_09/groundtruth.txt', 'endFrame': 1899, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_19_41_09', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_19_41_09/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_16_16_41/groundtruth.txt', 'endFrame': 1801, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_16_16_41', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_16_16_41/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_21_19_10/groundtruth.txt', 'endFrame': 2118, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_21_19_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_21_19_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_17_41_04/groundtruth.txt', 'endFrame': 3220, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_17_41_04', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_17_41_04/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_20_35_54/groundtruth.txt', 'endFrame': 3215, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_20_35_54', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_20_35_54/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_23_14_16/groundtruth.txt', 'endFrame': 1891, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_23_14_16', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_23_14_16/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_20_22_55/groundtruth.txt', 'endFrame': 1695, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_20_22_55', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_20_22_55/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_14_11_05_17/groundtruth.txt', 'endFrame': 2711, 'ext': 'jpg', 'name': 'dvSave-2022_10_14_11_05_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_14_11_05_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_18_49_30/groundtruth.txt', 'endFrame': 1933, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_18_49_30', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_18_49_30/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_20_34_35/groundtruth.txt', 'endFrame': 3888, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_20_34_35', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_20_34_35/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_19_43_03/groundtruth.txt', 'endFrame': 1821, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_19_43_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_19_43_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_10_44_05/groundtruth.txt', 'endFrame': 1850, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_10_44_05', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_10_44_05/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_14_20_43/groundtruth.txt', 'endFrame': 2048, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_14_20_43', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_14_20_43/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_47_47/groundtruth.txt', 'endFrame': 1711, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_47_47', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_47_47/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_20_45_02/groundtruth.txt', 'endFrame': 2086, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_20_45_02', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_20_45_02/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_58_25/groundtruth.txt', 'endFrame': 2118, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_58_25', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_58_25/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_31_10_25_58/groundtruth.txt', 'endFrame': 1841, 'ext': 'jpg', 'name': 'dvSave-2022_10_31_10_25_58', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_31_10_25_58/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_19_50_17/groundtruth.txt', 'endFrame': 1648, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_19_50_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_19_50_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_16_20_08/groundtruth.txt', 'endFrame': 1814, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_16_20_08', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_16_20_08/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_16_25_33/groundtruth.txt', 'endFrame': 2551, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_16_25_33', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_16_25_33/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_17_26_32/groundtruth.txt', 'endFrame': 2625, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_17_26_32', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_17_26_32/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_21_22_26/groundtruth.txt', 'endFrame': 2172, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_21_22_26', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_21_22_26/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_21_22_58_52/groundtruth.txt', 'endFrame': 2167, 'ext': 'jpg', 'name': 'dvSave-2022_10_21_22_58_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_21_22_58_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_14_11_12_25/groundtruth.txt', 'endFrame': 2389, 'ext': 'jpg', 'name': 'dvSave-2022_10_14_11_12_25', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_14_11_12_25/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_20_48_58/groundtruth.txt', 'endFrame': 2457, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_20_48_58', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_20_48_58/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_23_17_28/groundtruth.txt', 'endFrame': 1773, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_23_17_28', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_23_17_28/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_19_17_34/groundtruth.txt', 'endFrame': 2012, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_19_17_34', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_19_17_34/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_17_44_36/groundtruth.txt', 'endFrame': 3540, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_17_44_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_17_44_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_18_36_46/groundtruth.txt', 'endFrame': 2195, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_18_36_46', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_18_36_46/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_20_47_12/groundtruth.txt', 'endFrame': 1821, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_20_47_12', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_20_47_12/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_20_48_01/groundtruth.txt', 'endFrame': 2127, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_20_48_01', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_20_48_01/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_14_23_08/groundtruth.txt', 'endFrame': 1939, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_14_23_08', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_14_23_08/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_19_51_27/groundtruth.txt', 'endFrame': 2476, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_19_51_27', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_19_51_27/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_10_23/groundtruth.txt', 'endFrame': 2630, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_10_23', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_10_23/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_22_02_55/groundtruth.txt', 'endFrame': 2788, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_22_02_55', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_22_02_55/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_20_05_28/groundtruth.txt', 'endFrame': 1734, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_20_05_28', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_20_05_28/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_10_46_44/groundtruth.txt', 'endFrame': 1894, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_10_46_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_10_46_44/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_18_00_33/groundtruth.txt', 'endFrame': 2129, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_18_00_33', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_18_00_33/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_31_10_28_29/groundtruth.txt', 'endFrame': 1870, 'ext': 'jpg', 'name': 'dvSave-2022_10_31_10_28_29', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_31_10_28_29/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_20_23_52/groundtruth.txt', 'endFrame': 912, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_20_23_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_20_23_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_16_22_08/groundtruth.txt', 'endFrame': 1610, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_16_22_08', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_16_22_08/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_15_21_43/groundtruth.txt', 'endFrame': 2015, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_15_21_43', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_15_21_43/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_17_29_30/groundtruth.txt', 'endFrame': 2920, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_17_29_30', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_17_29_30/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_16_39_27/groundtruth.txt', 'endFrame': 2270, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_16_39_27', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_16_39_27/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_18_39_14/groundtruth.txt', 'endFrame': 2164, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_18_39_14', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_18_39_14/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_17_32_47/groundtruth.txt', 'endFrame': 2364, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_17_32_47', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_17_32_47/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_20_52_16/groundtruth.txt', 'endFrame': 1849, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_20_52_16', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_20_52_16/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_23_40_02/groundtruth.txt', 'endFrame': 1956, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_23_40_02', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_23_40_02/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_20_47_17/groundtruth.txt', 'endFrame': 2833, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_20_47_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_20_47_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_19_19_59/groundtruth.txt', 'endFrame': 2114, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_19_19_59', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_19_19_59/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_20_50_34/groundtruth.txt', 'endFrame': 1864, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_20_50_34', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_20_50_34/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_22_05_48/groundtruth.txt', 'endFrame': 1863, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_22_05_48', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_22_05_48/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_32_24/groundtruth.txt', 'endFrame': 2108, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_32_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_32_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_19_45_53/groundtruth.txt', 'endFrame': 1584, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_19_45_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_19_45_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_16_42_05/groundtruth.txt', 'endFrame': 2220, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_16_42_05', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_16_42_05/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_30_20_31_26/groundtruth.txt', 'endFrame': 2169, 'ext': 'jpg', 'name': 'dvSave-2022_10_30_20_31_26', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_30_20_31_26/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_12_21_01/groundtruth.txt', 'endFrame': 1852, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_12_21_01', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_12_21_01/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_16_51_07/groundtruth.txt', 'endFrame': 1930, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_16_51_07', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_16_51_07/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_19_22_10/groundtruth.txt', 'endFrame': 1970, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_19_22_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_19_22_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_21_08_53/groundtruth.txt', 'endFrame': 1935, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_21_08_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_21_08_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_21_20_37/groundtruth.txt', 'endFrame': 1880, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_21_20_37', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_21_20_37/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_30_20_33_36/groundtruth.txt', 'endFrame': 2309, 'ext': 'jpg', 'name': 'dvSave-2022_10_30_20_33_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_30_20_33_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_31_10_50_07/groundtruth.txt', 'endFrame': 2044, 'ext': 'jpg', 'name': 'dvSave-2022_10_31_10_50_07', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_31_10_50_07/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_31_10_52_10/groundtruth.txt', 'endFrame': 2018, 'ext': 'jpg', 'name': 'dvSave-2022_10_31_10_52_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_31_10_52_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_17_34_38/groundtruth.txt', 'endFrame': 2629, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_17_34_38', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_17_34_38/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_29_08/groundtruth.txt', 'endFrame': 1938, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_29_08', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_29_08/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_20_10_36/groundtruth.txt', 'endFrame': 1386, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_20_10_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_20_10_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_19_54_01/groundtruth.txt', 'endFrame': 2884, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_19_54_01', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_19_54_01/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_20_15_03/groundtruth.txt', 'endFrame': 1097, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_20_15_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_20_15_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_19_15_57/groundtruth.txt', 'endFrame': 2154, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_19_15_57', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_19_15_57/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_20_54_56/groundtruth.txt', 'endFrame': 2286, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_20_54_56', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_20_54_56/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_20_47_39/groundtruth.txt', 'endFrame': 1701, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_20_47_39', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_20_47_39/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_23_42_44/groundtruth.txt', 'endFrame': 1936, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_23_42_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_23_42_44/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_10_49_05/groundtruth.txt', 'endFrame': 1847, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_10_49_05', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_10_49_05/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_21_23_22/groundtruth.txt', 'endFrame': 1850, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_21_23_22', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_21_23_22/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_15_23_51/groundtruth.txt', 'endFrame': 2172, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_15_23_51', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_15_23_51/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_20_50_12/groundtruth.txt', 'endFrame': 3086, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_20_50_12', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_20_50_12/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_13_20/groundtruth.txt', 'endFrame': 2178, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_13_20', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_13_20/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_19_06_12/groundtruth.txt', 'endFrame': 3105, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_19_06_12', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_19_06_12/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_24_15/groundtruth.txt', 'endFrame': 2510, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_24_15', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_24_15/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_18_41_24/groundtruth.txt', 'endFrame': 2105, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_18_41_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_18_41_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_19_40_33/groundtruth.txt', 'endFrame': 2831, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_19_40_33', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_19_40_33/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_21_49_06/groundtruth.txt', 'endFrame': 1937, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_21_49_06', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_21_49_06/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_00_04_37/groundtruth.txt', 'endFrame': 2279, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_00_04_37', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_00_04_37/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_20_58_24/groundtruth.txt', 'endFrame': 1593, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_20_58_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_20_58_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_12_20_53_03/groundtruth.txt', 'endFrame': 1716, 'ext': 'jpg', 'name': 'dvSave-2022_10_12_20_53_03', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_12_20_53_03/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_19_28_44/groundtruth.txt', 'endFrame': 3291, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_19_28_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_19_28_44/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_19_21_51/groundtruth.txt', 'endFrame': 1597, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_19_21_51', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_19_21_51/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_17_36_42/groundtruth.txt', 'endFrame': 3085, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_17_36_42', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_17_36_42/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_22_08_39/groundtruth.txt', 'endFrame': 1909, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_22_08_39', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_22_08_39/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_21_48_17/groundtruth.txt', 'endFrame': 2052, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_21_48_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_21_48_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_19_18_27/groundtruth.txt', 'endFrame': 2183, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_19_18_27', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_19_18_27/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_23_44_53/groundtruth.txt', 'endFrame': 1911, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_23_44_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_23_44_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_12_23_22/groundtruth.txt', 'endFrame': 2013, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_12_23_22', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_12_23_22/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_16_53_46/groundtruth.txt', 'endFrame': 1505, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_16_53_46', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_16_53_46/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_16_57_27/groundtruth.txt', 'endFrame': 1901, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_16_57_27', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_16_57_27/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_19_43_09/groundtruth.txt', 'endFrame': 2130, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_19_43_09', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_19_43_09/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_18_58_42/groundtruth.txt', 'endFrame': 1762, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_18_58_42', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_18_58_42/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_15_47_10/groundtruth.txt', 'endFrame': 1977, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_15_47_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_15_47_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_21_32_16/groundtruth.txt', 'endFrame': 1857, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_21_32_16', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_21_32_16/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_16_40_05/groundtruth.txt', 'endFrame': 2099, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_16_40_05', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_16_40_05/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_20_53_14/groundtruth.txt', 'endFrame': 2277, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_20_53_14', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_20_53_14/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_31_24/groundtruth.txt', 'endFrame': 1213, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_31_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_31_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_19_25_33/groundtruth.txt', 'endFrame': 1907, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_19_25_33', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_19_25_33/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_19_50_53/groundtruth.txt', 'endFrame': 1797, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_19_50_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_19_50_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_16_44_19/groundtruth.txt', 'endFrame': 2296, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_16_44_19', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_16_44_19/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_30_20_51_41/groundtruth.txt', 'endFrame': 1549, 'ext': 'jpg', 'name': 'dvSave-2022_10_30_20_51_41', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_30_20_51_41/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_34_42/groundtruth.txt', 'endFrame': 2549, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_34_42', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_34_42/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_19_47_56/groundtruth.txt', 'endFrame': 2210, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_19_47_56', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_19_47_56/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_19_29_54/groundtruth.txt', 'endFrame': 1832, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_19_29_54', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_19_29_54/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_19_45_32/groundtruth.txt', 'endFrame': 1901, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_19_45_32', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_19_45_32/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_21_21_02/groundtruth.txt', 'endFrame': 3380, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_21_21_02', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_21_21_02/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_02_42/groundtruth.txt', 'endFrame': 2099, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_02_42', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_02_42/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_00_32/groundtruth.txt', 'endFrame': 2052, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_00_32', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_00_32/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_21_51_21/groundtruth.txt', 'endFrame': 1928, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_21_51_21', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_21_51_21/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_18_23_29_35/groundtruth.txt', 'endFrame': 1985, 'ext': 'jpg', 'name': 'dvSave-2022_10_18_23_29_35', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_18_23_29_35/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_18_15_02/groundtruth.txt', 'endFrame': 1830, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_18_15_02', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_18_15_02/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_20_37_20/groundtruth.txt', 'endFrame': 2362, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_20_37_20', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_20_37_20/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_11_21_02_23/groundtruth.txt', 'endFrame': 1645, 'ext': 'jpg', 'name': 'dvSave-2022_10_11_21_02_23', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_11_21_02_23/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_00_07_31/groundtruth.txt', 'endFrame': 2323, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_00_07_31', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_00_07_31/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_14_10_23_17/groundtruth.txt', 'endFrame': 2142, 'ext': 'jpg', 'name': 'dvSave-2022_10_14_10_23_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_14_10_23_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_21_51_15/groundtruth.txt', 'endFrame': 2267, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_21_51_15', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_21_51_15/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_30_20_53_17/groundtruth.txt', 'endFrame': 1761, 'ext': 'jpg', 'name': 'dvSave-2022_10_30_20_53_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_30_20_53_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_21_50_06/groundtruth.txt', 'endFrame': 2087, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_21_50_06', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_21_50_06/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_19_12_25_32/groundtruth.txt', 'endFrame': 1981, 'ext': 'jpg', 'name': 'dvSave-2022_10_19_12_25_32', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_19_12_25_32/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_23_16_32_50/groundtruth.txt', 'endFrame': 2533, 'ext': 'jpg', 'name': 'dvSave-2022_10_23_16_32_50', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_23_16_32_50/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_18_18_50/groundtruth.txt', 'endFrame': 2084, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_18_18_50', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_18_18_50/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_16_43_56/groundtruth.txt', 'endFrame': 2740, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_16_43_56', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_16_43_56/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_25_19_32_01/groundtruth.txt', 'endFrame': 3497, 'ext': 'jpg', 'name': 'dvSave-2022_10_25_19_32_01', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_25_19_32_01/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_28_19_53_10/groundtruth.txt', 'endFrame': 1994, 'ext': 'jpg', 'name': 'dvSave-2022_10_28_19_53_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_28_19_53_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_16_19_47_35/groundtruth.txt', 'endFrame': 2126, 'ext': 'jpg', 'name': 'dvSave-2022_10_16_19_47_35', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_16_19_47_35/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_29_17_04_47/groundtruth.txt', 'endFrame': 2140, 'ext': 'jpg', 'name': 'dvSave-2022_10_29_17_04_47', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_29_17_04_47/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_13_18_19_20/groundtruth.txt', 'endFrame': 1929, 'ext': 'jpg', 'name': 'dvSave-2022_10_13_18_19_20', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_13_18_19_20/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_15_19_31_55/groundtruth.txt', 'endFrame': 2137, 'ext': 'jpg', 'name': 'dvSave-2022_10_15_19_31_55', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_15_19_31_55/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_17_21_53_25/groundtruth.txt', 'endFrame': 2021, 'ext': 'jpg', 'name': 'dvSave-2022_10_17_21_53_25', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_17_21_53_25/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_18_23_31_42/groundtruth.txt', 'endFrame': 1877, 'ext': 'jpg', 'name': 'dvSave-2022_10_18_23_31_42', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_18_23_31_42/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_20_19_47_52/groundtruth.txt', 'endFrame': 2657, 'ext': 'jpg', 'name': 'dvSave-2022_10_20_19_47_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_20_19_47_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_22_21_24_22/groundtruth.txt', 'endFrame': 2639, 'ext': 'jpg', 'name': 'dvSave-2022_10_22_21_24_22', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_22_21_24_22/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2022_10_27_19_20_44/groundtruth.txt', 'endFrame': 2015, 'ext': 'jpg', 'name': 'dvSave-2022_10_27_19_20_44', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2022_10_27_19_20_44/inter1_stack_3008', 'startFrame': 0}]
        return sequence_info_list
