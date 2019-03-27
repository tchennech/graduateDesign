import cv2
import os
import os
import sys
import time
import pickle
import random
import numpy as np

class_num = 8
image_size = 32
img_channels = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#读取数据集 总共8类 32, 64, 128 , 256尺寸
#os opencv
def readImage(dir):
    totalImage =  []
    totalFlag = []
    for i in range(8):
        dirName = "10"+str(i+1)
        if i == 7:
            dirName = "102"
        realPath = dir + "/" + dirName
        for fileName in os.listdir(realPath):
            img = cv2.imread(realPath+"/"+fileName)
            if img is None:
                continue
            totalImage.append(img)
            totalFlag.append([i])
    #转成矩阵
    totalImage =  np.array(totalImage)
    totalFlag = np.array(totalFlag)
    return totalImage, totalFlag

#随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    #保存初始形状
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    #填充形状 上下左右都是 padding
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

#随机翻转
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        #随机1/2概率
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

#处理数据
def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    #修剪为[32, 32]
    batch = _random_crop(batch, [32, 32], 4)
    return batch