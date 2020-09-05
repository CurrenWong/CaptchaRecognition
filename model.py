#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


# In[2]:


# 设置参数
from params import *


# In[3]:


# 构建模型
def captcha_model():
    
    input_tensor = Input((height, width, channel))
    x = input_tensor
    for i in range(4):
        x = Conv2D(32*2**i, (3,3) ,activation='relu', data_format = 'channels_last')(x)
        x = Conv2D(32*2**i, (3,3) ,activation='relu', data_format = 'channels_last')(x)

        x = BatchNormalization(axis = -1)(x)
        x = MaxPooling2D( (2, 2), data_format = 'channels_last')(x)
    
    x = Flatten()(x)
    x = Dropout(0.7)(x)
    x = Dense(n_len * n_class,activation = 'softmax')(x)
    x = Reshape([n_len , n_class])(x)
    model = Model(inputs = (input_tensor), outputs = x)
    return model


# In[ ]:





# In[ ]:




