import cv2
import os
import os
import sys
import time
import pickle
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 读取数据集 总共8类 32, 64, 128 , 256尺寸
# os opencv 训练数据集
image_size = 32

def readImage(dir):
    totalImage = []
    totalFlag = []
    totalImageTemp = []
    totalFlagTemp = []
    testImgTemp = []
    testFlagTemp = []
    for i in range(8):
        dirName = "10"+str(i+1)
        if i == 7:
            dirName = "120"
        realPath = dir + "/" + dirName
        totalImageTemp = []
        totalFlagTemp = []
        for fileName in os.listdir(realPath):
            img = cv2.imread(realPath+"/"+fileName)
            if img is None:
                continue
            totalImageTemp.append(img)
            initData = [0., 0., 0., 0., 0., 0., 0., 0.]
            initData[i] = 1
            totalFlagTemp.append(initData)
        #打乱
        cc = list(zip(totalImageTemp, totalFlagTemp))
        random.shuffle(cc)
        totalImageTemp[:], totalFlagTemp[:] = zip(*cc)
        length = int(len(totalImageTemp) * 0.85)
        totalImage.extend(totalImageTemp[:length])
        totalFlag.extend(totalFlagTemp[:length])
        testImgTemp.extend(totalImageTemp[length:])
        testFlagTemp.extend(totalFlagTemp[length:])
    cc = list(zip(totalImage, totalFlag))
    random.shuffle(cc)
    totalImage[:], totalFlag[:] = zip(*cc)
    totalImage.extend(testImgTemp)
    totalFlag.extend(testFlagTemp)
    # 转成矩阵
    totalImage = np.array(totalImage)
    totalFlag = np.array(totalFlag)
    
    return totalImage, totalFlag


# 实际用到的数据集
def preperDatas(dir):
    totalImage = []
    label = []
    realPath = './data/' + dir
    for fileName in os.listdir(realPath):
        img = cv2.imread(realPath+"/"+fileName)
        if img is None:
            continue
        totalImage.append(img)
    # 转成矩阵
    totalImage = np.array(totalImage)
    return totalImage

# 随机裁剪


def _random_crop(batch, crop_shape, padding=None):
    # 保存初始形状
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    # 填充形状 上下左右都是 padding
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

# 随机翻转


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        # 随机1/2概率
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

# 处理数据


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    #修剪为[32, 32]
    batch = _random_crop(batch, [image_size, image_size], 4)
    return batch
