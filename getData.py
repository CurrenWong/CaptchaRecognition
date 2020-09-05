#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import numpy as np
import pandas as pd
import string
import tensorflow as tf


# In[2]:


# 设置参数
from params import *


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


# In[4]:


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




