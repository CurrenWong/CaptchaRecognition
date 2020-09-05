#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import os


# In[2]:


width, height, n_len, n_class, channel = 120, 80, 4, 62, 1
characters=string.digits+string.ascii_uppercase +  string.ascii_lowercase


# In[3]:


# 本机路径
baseDir = '.'
trainDir = os.path.join(baseDir, 'train')
valDir = os.path.join(baseDir, 'validate')
testDir = os.path.join(baseDir, 'test')
allDataDir = os.path.join(baseDir, 'allData')


# In[ ]:




