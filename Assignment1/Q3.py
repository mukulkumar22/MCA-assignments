#!/usr/bin/env python
# coding: utf-8

# In[127]:


import os
images = os.listdir("images")
import json
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from  skimage.transform import integral_image as integral_image
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix_det
from skimage.feature import hessian_matrix,peak_local_max


# In[129]:


for m in range(len(images)):
    image = io.imread("images/"+images[m])
    image = rgb2gray(image)
    image = integral_image(image)
    scales = np.linspace(1,8,20)
    hessians = []
    for s in scales:
        h = hessian_matrix_det(image, s) 
        hessians.append(h)
        
    h_cuboid  = np.dstack(hessians)
    blobs = peak_local_max(h_cuboid)
    d = [blobs[i] for i in range(1,len(blobs),500)]
    a = image
    Dict = []
    for i in d:
        Dict.append((i[0].item(),i[1].item(),i[2].item()))
    D = {images[m]:Dict}
    
    with open(images[m][:-4]+'surf.json', 'w') as fp:
        json.dump(D, fp)
    print("Saved",images[m][:-4]+"surf.json")
    
    
        

    
    
    
    


# In[ ]:





# In[109]:





# In[115]:


m = 0


# In[120]:





# In[122]:





# In[126]:





# In[ ]:





# In[ ]:




