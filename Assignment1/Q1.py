#!/usr/bin/env python
# coding: utf-8

# In[77]:


from PIL import Image
import os
import matplotlib.pyplot as plt
import PIL
import numpy as np


# In[78]:


images = os.listdir("images")


# In[80]:


all_images = {}


# In[81]:


path = "images/"


# In[82]:


all_colors = {i:0 for i in range(0,100)}


# In[83]:


k = 1


# In[84]:


for p in range(len(images)):
    print(images[p]  +" in process")
    a = Image.open(path + images[0])
    a = a.quantize(100)
    a = np.array(a)
    corr = {i: all_colors for i in range(0,100)}
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for u in range(-k,k+1):
                for v in range(-k,k+1):
                    if (i+u,j+v)!=(i,j):
                        try:
                            b = a[i+u][j+v]
                    
                            corr[a[i][j]][b] +=1
                            count+=1
                        except:
                            continue
    auto_corr = {i:0 for i in range(100)}
    for i in corr.keys():
        auto_corr[i] = corr[i][i]/(sum(corr[i].values())*8)
    all_images[images[p]] = auto_corr
    print(images[p]  +" done.")
    
    
                        
                        
    


# In[ ]:





# In[ ]:




