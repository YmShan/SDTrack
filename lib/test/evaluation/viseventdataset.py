import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class ViseventDataset(BaseDataset):
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
        self.base_path = self.env_settings.visevent_path
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

        return Sequence(sequence_info['name'], frames, 'visevent', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [{'anno_path': '00141_tank_outdoor2/groundtruth.txt', 'endFrame': 104, 'ext': 'jpg', 'name': '00141_tank_outdoor2', 'nz': 4, 'object_class': 'object', 'path': '00141_tank_outdoor2/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00147_tank_outdoor2/groundtruth.txt', 'endFrame': 103, 'ext': 'jpg', 'name': '00147_tank_outdoor2', 'nz': 4, 'object_class': 'object', 'path': '00147_tank_outdoor2/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00197_driving_outdoor3/groundtruth.txt', 'endFrame': 104, 'ext': 'jpg', 'name': '00197_driving_outdoor3', 'nz': 4, 'object_class': 'object', 'path': '00197_driving_outdoor3/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00236_tennis_outdoor4/groundtruth.txt', 'endFrame': 104, 'ext': 'jpg', 'name': '00236_tennis_outdoor4', 'nz': 4, 'object_class': 'object', 'path': '00236_tennis_outdoor4/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00241_tennis_outdoor4/groundtruth.txt', 'endFrame': 103, 'ext': 'jpg', 'name': '00241_tennis_outdoor4', 'nz': 4, 'object_class': 'object', 'path': '00241_tennis_outdoor4/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00340_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00340_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00340_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00345_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00345_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00345_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00351_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00351_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00351_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00355_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00355_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00355_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00385_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00385_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00385_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00398_UAV_outdoor6/groundtruth.txt', 'endFrame': 128, 'ext': 'jpg', 'name': '00398_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00398_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00404_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00404_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00404_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00406_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00406_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00406_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00408_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00408_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00408_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00410_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00410_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00410_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00413_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00413_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00413_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00416_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00416_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00416_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00421_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00421_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00421_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00423_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00423_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00423_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00425_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00425_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00425_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00430_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00430_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00430_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00432_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00432_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00432_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00435_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00435_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00435_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00437_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00437_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00437_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00439_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00439_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00439_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00442_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00442_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00442_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00445_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00445_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00445_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00447_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00447_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00447_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00449_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00449_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00449_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00451_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00451_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00451_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00453_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00453_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00453_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00458_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00458_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00458_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00464_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00464_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00464_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00466_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00466_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00466_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00471_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00471_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00471_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00473_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00473_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00473_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00490_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00490_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00490_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00503_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00503_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00503_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00506_person_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00506_person_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00506_person_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00508_person_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00508_person_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00508_person_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00510_person_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00510_person_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00510_person_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00511_person_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00511_person_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00511_person_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00514_person_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00514_person_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00514_person_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dightNUM_001/groundtruth.txt', 'endFrame': 534, 'ext': 'jpg', 'name': 'dightNUM_001', 'nz': 4, 'object_class': 'object', 'path': 'dightNUM_001/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_04_20_41_53/groundtruth.txt', 'endFrame': 1118, 'ext': 'jpg', 'name': 'dvSave-2021_02_04_20_41_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_04_20_41_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_04_20_49_43/groundtruth.txt', 'endFrame': 1776, 'ext': 'jpg', 'name': 'dvSave-2021_02_04_20_49_43', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_04_20_49_43/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_04_20_56_55/groundtruth.txt', 'endFrame': 115, 'ext': 'jpg', 'name': 'dvSave-2021_02_04_20_56_55', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_04_20_56_55/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00370_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00370_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00370_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00433_UAV_outdoor6/groundtruth.txt', 'endFrame': 88, 'ext': 'jpg', 'name': '00433_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00433_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': '00478_UAV_outdoor6/groundtruth.txt', 'endFrame': 87, 'ext': 'jpg', 'name': '00478_UAV_outdoor6', 'nz': 4, 'object_class': 'object', 'path': '00478_UAV_outdoor6/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_21_53_car/groundtruth.txt', 'endFrame': 45, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_21_53_car', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_21_53_car/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_10_11_59_paperClips/groundtruth.txt', 'endFrame': 1033, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_10_11_59_paperClips', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_10_11_59_paperClips/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_21_41_personFootball/groundtruth.txt', 'endFrame': 571, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_21_41_personFootball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_21_41_personFootball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_57_54_personFootball/groundtruth.txt', 'endFrame': 973, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_57_54_personFootball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_57_54_personFootball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_14_17_00_48/groundtruth.txt', 'endFrame': 961, 'ext': 'jpg', 'name': 'dvSave-2021_02_14_17_00_48', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_14_17_00_48/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_12_44_27_chicken/groundtruth.txt', 'endFrame': 205, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_12_44_27_chicken', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_12_44_27_chicken/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_12_45_redcar/groundtruth.txt', 'endFrame': 127, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_12_45_redcar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_12_45_redcar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_29_37/groundtruth.txt', 'endFrame': 1500, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_29_37', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_29_37/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0015/groundtruth.txt', 'endFrame': 366, 'ext': 'jpg', 'name': 'video_0015', 'nz': 4, 'object_class': 'object', 'path': 'video_0015/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_04_21_18_52/groundtruth.txt', 'endFrame': 311, 'ext': 'jpg', 'name': 'dvSave-2021_02_04_21_18_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_04_21_18_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_04_21_20_22/groundtruth.txt', 'endFrame': 256, 'ext': 'jpg', 'name': 'dvSave-2021_02_04_21_20_22', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_04_21_20_22/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_04_21_21_24/groundtruth.txt', 'endFrame': 219, 'ext': 'jpg', 'name': 'dvSave-2021_02_04_21_21_24', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_04_21_21_24/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_08_56_18_windowPattern/groundtruth.txt', 'endFrame': 347, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_08_56_18_windowPattern', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_08_56_18_windowPattern/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_08_56_40_windowPattern2/groundtruth.txt', 'endFrame': 177, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_08_56_40_windowPattern2', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_08_56_40_windowPattern2/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dydrant_001/groundtruth.txt', 'endFrame': 502, 'ext': 'jpg', 'name': 'dydrant_001', 'nz': 4, 'object_class': 'object', 'path': 'dydrant_001/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_16_35_car/groundtruth.txt', 'endFrame': 168, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_16_35_car', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_16_35_car/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_24_26_Pedestrian1/groundtruth.txt', 'endFrame': 278, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_24_26_Pedestrian1', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_24_26_Pedestrian1/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_35_08_Pedestrian/groundtruth.txt', 'endFrame': 829, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_35_08_Pedestrian', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_35_08_Pedestrian/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_36_15_Pedestrian/groundtruth.txt', 'endFrame': 488, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_36_15_Pedestrian', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_36_15_Pedestrian/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_36_44_Pedestrian/groundtruth.txt', 'endFrame': 196, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_36_44_Pedestrian', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_36_44_Pedestrian/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_09_58_27_DigitAI/groundtruth.txt', 'endFrame': 941, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_09_58_27_DigitAI', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_09_58_27_DigitAI/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_10_03_17_GreenPlant/groundtruth.txt', 'endFrame': 2584, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_10_03_17_GreenPlant', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_10_03_17_GreenPlant/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_10_05_38_phone/groundtruth.txt', 'endFrame': 2298, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_10_05_38_phone', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_10_05_38_phone/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_10_09_04_bottle/groundtruth.txt', 'endFrame': 609, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_10_09_04_bottle', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_10_09_04_bottle/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_10_14_17_paperClip/groundtruth.txt', 'endFrame': 1514, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_10_14_17_paperClip', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_10_14_17_paperClip/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_10_17_16_paperClips/groundtruth.txt', 'endFrame': 1741, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_10_17_16_paperClips', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_10_17_16_paperClips/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_15_08_41_flag/groundtruth.txt', 'endFrame': 613, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_15_08_41_flag', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_15_08_41_flag/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_15_12_44_car/groundtruth.txt', 'endFrame': 190, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_15_12_44_car', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_15_12_44_car/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_15_14_26_blackcar/groundtruth.txt', 'endFrame': 784, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_15_14_26_blackcar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_15_14_26_blackcar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_15_15_36_redcar/groundtruth.txt', 'endFrame': 139, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_15_15_36_redcar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_15_15_36_redcar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_15_17_48_whitecar/groundtruth.txt', 'endFrame': 118, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_15_17_48_whitecar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_15_17_48_whitecar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_15_18_36_redcar/groundtruth.txt', 'endFrame': 195, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_15_18_36_redcar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_15_18_36_redcar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_15_20_whitecar/groundtruth.txt', 'endFrame': 667, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_15_20_whitecar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_15_20_whitecar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_16_26_whitecar/groundtruth.txt', 'endFrame': 505, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_16_26_whitecar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_16_26_whitecar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_20_28_personFootball/groundtruth.txt', 'endFrame': 520, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_20_28_personFootball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_20_28_personFootball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_23_26_personFootball/groundtruth.txt', 'endFrame': 724, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_23_26_personFootball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_23_26_personFootball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_27_53_personFootball/groundtruth.txt', 'endFrame': 655, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_27_53_personFootball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_27_53_personFootball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_31_03_personBasketball/groundtruth.txt', 'endFrame': 523, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_31_03_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_31_03_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_33_01_personBasketball/groundtruth.txt', 'endFrame': 912, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_33_01_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_33_01_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_34_58_personBasketball/groundtruth.txt', 'endFrame': 648, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_34_58_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_34_58_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_36_49_personBasketball/groundtruth.txt', 'endFrame': 730, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_36_49_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_36_49_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_41_45_personBasketball/groundtruth.txt', 'endFrame': 647, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_41_45_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_41_45_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_45_17_personBasketball/groundtruth.txt', 'endFrame': 614, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_45_17_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_45_17_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_47_49_personBasketball/groundtruth.txt', 'endFrame': 1088, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_47_49_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_47_49_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_49_51_personBasketball/groundtruth.txt', 'endFrame': 533, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_49_51_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_49_51_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_51_05_personBasketball/groundtruth.txt', 'endFrame': 786, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_51_05_personBasketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_51_05_personBasketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_06_17_53_39_personFootball/groundtruth.txt', 'endFrame': 1135, 'ext': 'jpg', 'name': 'dvSave-2021_02_06_17_53_39_personFootball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_06_17_53_39_personFootball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_08_21_05_56_motor/groundtruth.txt', 'endFrame': 66, 'ext': 'jpg', 'name': 'dvSave-2021_02_08_21_05_56_motor', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_08_21_05_56_motor/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_08_21_07_52/groundtruth.txt', 'endFrame': 142, 'ext': 'jpg', 'name': 'dvSave-2021_02_08_21_07_52', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_08_21_07_52/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_12_13_38_26/groundtruth.txt', 'endFrame': 2254, 'ext': 'jpg', 'name': 'dvSave-2021_02_12_13_38_26', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_12_13_38_26/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_12_13_39_56/groundtruth.txt', 'endFrame': 1587, 'ext': 'jpg', 'name': 'dvSave-2021_02_12_13_39_56', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_12_13_39_56/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_12_13_43_54/groundtruth.txt', 'endFrame': 1197, 'ext': 'jpg', 'name': 'dvSave-2021_02_12_13_43_54', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_12_13_43_54/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_12_13_46_18/groundtruth.txt', 'endFrame': 1098, 'ext': 'jpg', 'name': 'dvSave-2021_02_12_13_46_18', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_12_13_46_18/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_12_13_56_29/groundtruth.txt', 'endFrame': 915, 'ext': 'jpg', 'name': 'dvSave-2021_02_12_13_56_29', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_12_13_56_29/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_14_16_21_40/groundtruth.txt', 'endFrame': 563, 'ext': 'jpg', 'name': 'dvSave-2021_02_14_16_21_40', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_14_16_21_40/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_14_16_22_06/groundtruth.txt', 'endFrame': 552, 'ext': 'jpg', 'name': 'dvSave-2021_02_14_16_22_06', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_14_16_22_06/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_14_16_53_15_flag/groundtruth.txt', 'endFrame': 1199, 'ext': 'jpg', 'name': 'dvSave-2021_02_14_16_53_15_flag', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_14_16_53_15_flag/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_14_16_56_01_house/groundtruth.txt', 'endFrame': 371, 'ext': 'jpg', 'name': 'dvSave-2021_02_14_16_56_01_house', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_14_16_56_01_house/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_12_19_basketball/groundtruth.txt', 'endFrame': 1602, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_12_19_basketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_12_19_basketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_14_18_chicken/groundtruth.txt', 'endFrame': 1611, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_14_18_chicken', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_14_18_chicken/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_22_23_basketball/groundtruth.txt', 'endFrame': 377, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_22_23_basketball', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_22_23_basketball/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_22_23_boyhead/groundtruth.txt', 'endFrame': 377, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_22_23_boyhead', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_22_23_boyhead/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_23_05_basketall/groundtruth.txt', 'endFrame': 1395, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_23_05_basketall', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_23_05_basketall/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_23_05_boyhead/groundtruth.txt', 'endFrame': 1395, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_23_05_boyhead', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_23_05_boyhead/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_10_26_11_chicken/groundtruth.txt', 'endFrame': 702, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_10_26_11_chicken', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_10_26_11_chicken/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_12_45_02_Duck/groundtruth.txt', 'endFrame': 462, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_12_45_02_Duck', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_12_45_02_Duck/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_12_53_54_personHead/groundtruth.txt', 'endFrame': 459, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_12_53_54_personHead', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_12_53_54_personHead/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_01_16_Duck/groundtruth.txt', 'endFrame': 296, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_01_16_Duck', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_01_16_Duck/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_02_21_Chicken/groundtruth.txt', 'endFrame': 503, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_02_21_Chicken', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_02_21_Chicken/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_04_57_Duck/groundtruth.txt', 'endFrame': 541, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_04_57_Duck', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_04_57_Duck/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_05_43_Chicken/groundtruth.txt', 'endFrame': 205, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_05_43_Chicken', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_05_43_Chicken/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_08_12_blackcar/groundtruth.txt', 'endFrame': 271, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_08_12_blackcar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_08_12_blackcar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_09_09_person/groundtruth.txt', 'endFrame': 855, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_09_09_person', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_09_09_person/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_10_54_person/groundtruth.txt', 'endFrame': 244, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_10_54_person', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_10_54_person/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_13_44_whitecar/groundtruth.txt', 'endFrame': 105, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_13_44_whitecar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_13_44_whitecar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_14_18_blackcar/groundtruth.txt', 'endFrame': 105, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_14_18_blackcar', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_14_18_blackcar/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_24_03_girlhead/groundtruth.txt', 'endFrame': 204, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_24_03_girlhead', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_24_03_girlhead/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_24_49_girlhead/groundtruth.txt', 'endFrame': 396, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_24_49_girlhead', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_24_49_girlhead/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_27_20_bottle/groundtruth.txt', 'endFrame': 882, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_27_20_bottle', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_27_20_bottle/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_13_28_20_cash/groundtruth.txt', 'endFrame': 738, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_13_28_20_cash', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_13_28_20_cash/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_23_51_36/groundtruth.txt', 'endFrame': 845, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_23_51_36', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_23_51_36/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_23_54_17/groundtruth.txt', 'endFrame': 355, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_23_54_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_23_54_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_15_23_56_17/groundtruth.txt', 'endFrame': 667, 'ext': 'jpg', 'name': 'dvSave-2021_02_15_23_56_17', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_15_23_56_17/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_07_38/groundtruth.txt', 'endFrame': 960, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_07_38', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_07_38/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_12_18/groundtruth.txt', 'endFrame': 1525, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_12_18', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_12_18/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_15_53/groundtruth.txt', 'endFrame': 1552, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_15_53', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_15_53/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_20_20/groundtruth.txt', 'endFrame': 984, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_20_20', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_20_20/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_23_10/groundtruth.txt', 'endFrame': 1944, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_23_10', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_23_10/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_34_11/groundtruth.txt', 'endFrame': 1503, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_34_11', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_34_11/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_38_25/groundtruth.txt', 'endFrame': 744, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_38_25', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_38_25/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'dvSave-2021_02_16_17_42_50/groundtruth.txt', 'endFrame': 877, 'ext': 'jpg', 'name': 'dvSave-2021_02_16_17_42_50', 'nz': 4, 'object_class': 'object', 'path': 'dvSave-2021_02_16_17_42_50/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'roadLight_001/groundtruth.txt', 'endFrame': 374, 'ext': 'jpg', 'name': 'roadLight_001', 'nz': 4, 'object_class': 'object', 'path': 'roadLight_001/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_001/groundtruth.txt', 'endFrame': 1013, 'ext': 'jpg', 'name': 'tennis_long_001', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_001/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_002/groundtruth.txt', 'endFrame': 901, 'ext': 'jpg', 'name': 'tennis_long_002', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_002/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_003/groundtruth.txt', 'endFrame': 1264, 'ext': 'jpg', 'name': 'tennis_long_003', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_003/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_004/groundtruth.txt', 'endFrame': 906, 'ext': 'jpg', 'name': 'tennis_long_004', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_004/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_005/groundtruth.txt', 'endFrame': 1586, 'ext': 'jpg', 'name': 'tennis_long_005', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_005/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_006/groundtruth.txt', 'endFrame': 937, 'ext': 'jpg', 'name': 'tennis_long_006', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_006/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'tennis_long_007/groundtruth.txt', 'endFrame': 1553, 'ext': 'jpg', 'name': 'tennis_long_007', 'nz': 4, 'object_class': 'object', 'path': 'tennis_long_007/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'UAV_long_001/groundtruth.txt', 'endFrame': 2079, 'ext': 'jpg', 'name': 'UAV_long_001', 'nz': 4, 'object_class': 'object', 'path': 'UAV_long_001/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0004/groundtruth.txt', 'endFrame': 1400, 'ext': 'jpg', 'name': 'video_0004', 'nz': 4, 'object_class': 'object', 'path': 'video_0004/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0005/groundtruth.txt', 'endFrame': 693, 'ext': 'jpg', 'name': 'video_0005', 'nz': 4, 'object_class': 'object', 'path': 'video_0005/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0008/groundtruth.txt', 'endFrame': 241, 'ext': 'jpg', 'name': 'video_0008', 'nz': 4, 'object_class': 'object', 'path': 'video_0008/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0018/groundtruth.txt', 'endFrame': 387, 'ext': 'jpg', 'name': 'video_0018', 'nz': 4, 'object_class': 'object', 'path': 'video_0018/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0021/groundtruth.txt', 'endFrame': 1108, 'ext': 'jpg', 'name': 'video_0021', 'nz': 4, 'object_class': 'object', 'path': 'video_0021/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0026/groundtruth.txt', 'endFrame': 808, 'ext': 'jpg', 'name': 'video_0026', 'nz': 4, 'object_class': 'object', 'path': 'video_0026/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0029/groundtruth.txt', 'endFrame': 1412, 'ext': 'jpg', 'name': 'video_0029', 'nz': 4, 'object_class': 'object', 'path': 'video_0029/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0032/groundtruth.txt', 'endFrame': 2925, 'ext': 'jpg', 'name': 'video_0032', 'nz': 4, 'object_class': 'object', 'path': 'video_0032/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0039/groundtruth.txt', 'endFrame': 269, 'ext': 'jpg', 'name': 'video_0039', 'nz': 4, 'object_class': 'object', 'path': 'video_0039/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0045/groundtruth.txt', 'endFrame': 218, 'ext': 'jpg', 'name': 'video_0045', 'nz': 4, 'object_class': 'object', 'path': 'video_0045/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0049/groundtruth.txt', 'endFrame': 544, 'ext': 'jpg', 'name': 'video_0049', 'nz': 4, 'object_class': 'object', 'path': 'video_0049/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0050/groundtruth.txt', 'endFrame': 1179, 'ext': 'jpg', 'name': 'video_0050', 'nz': 4, 'object_class': 'object', 'path': 'video_0050/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0054/groundtruth.txt', 'endFrame': 1261, 'ext': 'jpg', 'name': 'video_0054', 'nz': 4, 'object_class': 'object', 'path': 'video_0054/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0056/groundtruth.txt', 'endFrame': 1847, 'ext': 'jpg', 'name': 'video_0056', 'nz': 4, 'object_class': 'object', 'path': 'video_0056/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0058/groundtruth.txt', 'endFrame': 953, 'ext': 'jpg', 'name': 'video_0058', 'nz': 4, 'object_class': 'object', 'path': 'video_0058/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0060/groundtruth.txt', 'endFrame': 1032, 'ext': 'jpg', 'name': 'video_0060', 'nz': 4, 'object_class': 'object', 'path': 'video_0060/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0064/groundtruth.txt', 'endFrame': 578, 'ext': 'jpg', 'name': 'video_0064', 'nz': 4, 'object_class': 'object', 'path': 'video_0064/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0067/groundtruth.txt', 'endFrame': 1553, 'ext': 'jpg', 'name': 'video_0067', 'nz': 4, 'object_class': 'object', 'path': 'video_0067/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0070/groundtruth.txt', 'endFrame': 630, 'ext': 'jpg', 'name': 'video_0070', 'nz': 4, 'object_class': 'object', 'path': 'video_0070/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0073/groundtruth.txt', 'endFrame': 488, 'ext': 'jpg', 'name': 'video_0073', 'nz': 4, 'object_class': 'object', 'path': 'video_0073/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0076/groundtruth.txt', 'endFrame': 119, 'ext': 'jpg', 'name': 'video_0076', 'nz': 4, 'object_class': 'object', 'path': 'video_0076/inter1_stack_3008', 'startFrame': 0},
{'anno_path': 'video_0079/groundtruth.txt', 'endFrame': 1124, 'ext': 'jpg', 'name': 'video_0079', 'nz': 4, 'object_class': 'object', 'path': 'video_0079/inter1_stack_3008', 'startFrame': 0}
        ]
        return sequence_info_list
