#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import os


# In[2]:


images = os.listdir("images")
path = "images/"


# In[3]:


def detect_blobs(image,outputs,sigma,k):
    
    """reference :https://projectsflix.com/opencv/laplacian-blob-detector-using-python/"""
    
    positions = []
    h = image.shape[0]
    w = image.shape[1]
    
    for i in range(1,h):
        for j in range(1,w):
            se = outputs[:,i-1:i+2,j-1:j+2]
            result = np.amax(se)
            if result >= 0.5:
                z,x,y = np.unravel_index(se.argmax(),se.shape)
                positions.append((i+x-1,j+y-1,k**z*sigma))
                
    return positions
            


# In[ ]:


for m in range(len(images)):
    print(images[m])
    
    a = cv2.imread(path+images[m],0)
    Log_images = []
    blur = cv2.GaussianBlur(a,(5,5),0)
    for i in range(1,6):
        s = cv2.Laplacian(blur,5,scale = 1.414**i)
        Log_images.append(s)
    log = np.array(Log_images)
    for i in range(1,6):
        s = cv2.Laplacian(blur,5,scale = 1.414**i)
        Log_images.append(s)
    log = log/255
    d =detect_blobs(a,log,1,k =1.414)
    
    Dict = []
    for i in d:
        Dict.append((i[0].item(),i[1].item(),i[2].item()))
    D = {images[m]:Dict}
    with open(images[m]+'.json', 'w') as fp:
        json.dump(D, fp)
    print("Saved",images[m]+".json")
    

    

