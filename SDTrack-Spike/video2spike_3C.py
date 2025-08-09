import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import torch

def save_1C_img(img, root):
    two_channel_img = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    two_channel_img[:, :, 0] = img
    cv2.imwrite(root, two_channel_img)

def save_2C_img(pos_img, neg_img, null_img, root):
    # 创建一个两通道的图像
    two_channel_img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), dtype=np.uint8)
    two_channel_img[:, :, 0] = pos_img
    two_channel_img[:, :, 1] = neg_img
    two_channel_img[:, :, 2] = null_img
    cv2.imwrite(root, two_channel_img)

def Video2Spike(data_list, save_name, T_num, threshold, noise, decay):
    for dataset_path in sorted(data_list):
        image_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path)])
        with Image.open(image_files[0]) as img:
            W, H = img.size
        L = len(image_files)
        pic_shape = (H, W, 3)

        base_dir = os.path.dirname(dataset_path)
        save_dir = os.path.join(base_dir, save_name)
        if(os.path.isdir(save_dir)):
            continue
        os.makedirs(save_dir, exist_ok=True)

        image_arrays = []
        for path in image_files:
            img = Image.open(path).convert('L')
            img_array = np.array(img)
            img_array = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
            image_arrays.append(img_array)

        Spike_Matrix = np.zeros((L, H, W, 3), dtype=np.uint8)
        if(noise):
            U = np.random.random(size=pic_shape) * threshold
        else:
            U = np.full(pic_shape, 0.0, dtype=np.float64)
        Thr = np.ones_like(pic_shape).astype(np.float32) * threshold
        count = 0
        for t in tqdm(range(L), desc=f"Processing {dataset_path}"):
            img = image_arrays[t]
            I = img / 255.0
            U_mem = []

            for i in range(T_num):
                #spike1
                U = I + U
                S = (U - Thr) >= 0
                U = U * (1 - S)
                U = U * decay
                U_mem.append(U)
                count = count + S.sum()
                #spike2
                # U = I + U
                # S = (U - Thr) >= 0
                # S = S.astype(np.int32)
                # U = U - threshold * S
                # U = U * decay
                # U_mem.append(S)
            U = sum(U_mem)
            Spike_Matrix[t] = U
            file_path = save_dir + '/' + str(t).zfill(4) + '.png'
            save_2C_img(Spike_Matrix[t,:,:,0], Spike_Matrix[t,:,:,1], Spike_Matrix[t,:,:,2], file_path)
        print(count)
    return 0

def Stack_Video(data_path, save_name, T_num, threshold, noise, decay):
    train_path = os.path.join(data_path, 'train')
    train_list = []
    for item in os.listdir(train_path):
        subdir_path = os.path.join(train_path, item)
        img_path = os.path.join(subdir_path, "img")
        # 判断是否是目录并且其下有img子目录
        if os.path.isdir(subdir_path) and os.path.isdir(img_path):
            train_list.append(img_path)
    test_path = os.path.join(data_path, 'test')
    test_list = []
    for item in os.listdir(test_path):
        subdir_path = os.path.join(test_path, item)
        img_path = os.path.join(subdir_path, "img")
        # 判断是否是目录并且其下有img子目录
        if os.path.isdir(subdir_path) and os.path.isdir(img_path):
            train_list.append(img_path)
    Video2Spike(train_list, save_name, T_num, threshold, noise, decay)
    Video2Spike(test_list, save_name, T_num, threshold, noise, decay)
    return 0

if __name__ == '__main__':
    Stack_Video(data_path='/data1/dataset/FE108', save_name='spike_thresh_2.5_decay_0.25', T_num=255, threshold=2.5, noise = True, decay=0.25)
    Stack_Video(data_path='/data1/dataset/VisEvent', save_name='spike_thresh_2.5_decay_0.50', T_num=255, threshold=2.5, noise = True,decay=0.25)
    Stack_Video(data_path='/data/dataset/COESOT_dataset', save_name='spike_thresh_2.5_decay_0.25', T_num=255, threshold=2.5, noise = True,decay=0.25)
