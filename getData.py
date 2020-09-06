#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import string
import tensorflow as tf


# In[2]:


# 设置参数
from params import *


# ## Dataset

# In[3]:



def getPath(path):
    '''
    获取路径
    '''
    for root, dirs, files in os.walk(trainDir):
        # 去除隐藏文件
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        # 添加系统路径
        files = pd.Series(files).apply(lambda x: os.path.join(trainDir, x)).values
    return files

def getLabel(filesPath):
    #  获取标签
    labels = np.zeros((len(filesPath), n_len , n_class), dtype= np.uint8)
    for i in range(len(filesPath)):
        #获取文件名
        name =  os.path.splitext(filesPath[i])[0].split('_')
        num, content = name[0], name[1]
        # 把标记赋值给张量
        for j, ch in enumerate(content):
            labels[i][j, :] = 0
            labels[i][j, characters.find(ch)] = 1
    return labels


# In[4]:


def load_and_preprocess_image(filePath):
    # 读取图片
    image = tf.io.read_file(filePath)
    # 将png格式的图片解码，得到一个张量（一维的矩阵）
    image = tf.image.decode_png(image, channels=1)
    # 调整大小
    image = tf.image.resize(image, [height, width])
    # 对每个像素点的RGB值做归一化处理
    image /= 255.0
    return image


# In[5]:


def getDataset(dirPath):
    # 获取图片
    filesPath = getPath(dirPath)
    # 获取标签
    labels = getLabel(filesPath)
    # 构建图片路径的“dataset”
    dataset = tf.data.Dataset.from_tensor_slices(filesPath)
    # 使用AUTOTUNE自动调节管道参数
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # 处理图片
    image_ds = dataset.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
    # 构建类标数据的“dataset”
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    # 将图片和类标压缩为（图片，类标）对
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # 形成batch
    image_label_ds = image_label_ds.batch(batch_size)
    # 让数据集重复多次
    image_label_ds = image_label_ds.repeat()
    # 通过“prefetch”方法让模型的训练和每个batch数据集的加载并行
    image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    return image_label_ds


# ## 数据生成器

# In[8]:


def gen(path = trainDir, batch_size  = 32):
    '''
    获取训练数据/验证数据
    '''
    X = np.zeros((batch_size, height, width, channel), dtype= np.float16)
    y = np.zeros((batch_size, n_len , n_class), dtype= np.uint8)
    # 遍历目录
    for root, dirs, files in os.walk(path):
        # 去除隐藏文件
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        # 设置起始指针
        pointer = 0
        while(True):
            # 若指针超过文件数量，从头开始
            if pointer + batch_size >= len(files):
                pointer = 0
            # 遍历文件名
            for i in range(batch_size):

                file = files[pointer + i]

                #获取文件名
                name =  os.path.splitext(file)[0].split('_')
                num, content = name[0], name[1]

                #生成读取路径
                readPath = os.path.join(path, file)
                # 读取图片
                imgBuffer = cv2.imread(readPath, 0)
                # 改变图片大小
                imgBuffer = cv2.resize(imgBuffer, (width, height))

                # 二值化
                #  t, imgBuffer = cv2.threshold(imgBuffer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 归一化
                minX = imgBuffer.min()
                imgBuffer = imgBuffer - minX
                maxX = max(1, imgBuffer.max())
                imgBuffer = np.array(imgBuffer / maxX, np.float16)
                
                # 改变图片维度，适应模型输入
                imgBuffer = np.expand_dims(imgBuffer, axis = 2)

                # 把图像赋值给张量
                X[i] = imgBuffer
                
                # 把标记赋值给张量
                for j, ch in enumerate(content):
                    y[i][j, :] = 0
                    y[i][j, characters.find(ch)] = 1

            # 指针指向下一个batch
            pointer += batch_size
            
            # 输出
            yield X, y


# In[9]:


def test_gen(path = testDir, batch_size  = 1):
    
    '''
    获取测试数据
    '''
    
    X = np.zeros((batch_size, height, width, channel), dtype= np.float16)
    
    # 遍历目录
    for root, dirs, files in os.walk(path):
        # 去除隐藏文件
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        # 设置起始指针
        pointer = 0
        while(True):
            # 若指针超过文件数量，从头开始
            if pointer + batch_size >= len(files):
                pointer = 0
            # 遍历文件名
            for i in range(batch_size):

                file = files[pointer + i]

                #获取文件名
                num =  os.path.splitext(file)[0]

                #生成读取路径
                readPath = os.path.join(path, file)
                # 读取图片
                imgBuffer = cv2.imread(readPath, 0)
                # 改变图片大小
                imgBuffer = cv2.resize(imgBuffer, (width, height))

                # 二值化
                #  t, imgBuffer = cv2.threshold(imgBuffer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 归一化
                minX = imgBuffer.min()
                imgBuffer = imgBuffer - minX
                maxX = max(1, imgBuffer.max())
                # 将对比度拉伸到0-255范围内
                imgBuffer = np.array(imgBuffer / maxX, np.float16)
                # 改变图片维度，适应模型输入
                imgBuffer = np.expand_dims(imgBuffer, axis = 2)

                # 把图像赋值给张量
                X[i] = imgBuffer
                
            # 指针指向下一个batch
            pointer += batch_size
        
            # 输出
            yield X, num


# In[ ]:




