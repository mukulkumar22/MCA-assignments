#!/usr/bin/env python
# coding: utf-8

# In[142]:


import os
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import Audio
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.svm import SVC
import numpy as np
import pandas as pd


# In[143]:


def get_indi_dft(data,n):
    L = len(data)
    k = [i for i in range(L)]
    k = np.array(k)
    coeff = (-1j*2*np.pi*k*n)/L
    x = data*np.exp(coeff)
    x = np.sum(x)
    x = x/L
    x = np.abs(x)
    
    return x
    


# In[144]:


def get_DFT(data):
    dft = []
    for i in range(len(data)//2):
        dft.append(get_indi_dft(data,i)*2)
    return dft


# In[145]:


def spectrogram(data,windowsize,overlap,plot = False):
    startinds = [0 +(windowsize-overlap)*i for i in range(len(data)) if (windowsize-overlap)*i <len(data)-windowsize]
    #startinds = [i for i in startinds if i<len(data)-windowsize]
    dfts = []
    for i in startinds:
        window = get_DFT(data[i:i+windowsize])
        dfts.append(window)
    spectro = np.array(dfts).T
    spectro = 10*np.log10(spectro)
    
    if plot:
        
        plt.imshow(spectro)
    
    return spectro


# In[162]:


path_training = 'Dataset/training/'
path_validation = 'Dataset/validation/'


# In[163]:


classes = os.listdir('Dataset/validation')


# In[164]:


def create_dataset(sound_class):
    dataset = []
    files = os.listdir(path_validation+sound_class)
    for i in files:
        
        sr,data = wavfile.read(path_validation+sound_class+"/"+i)
        dataset.append(spectrogram(data,256,64))
    return dataset
    


# In[165]:


dataset = []
labels = []
for i in classes:
    print(i)
    data = create_dataset(i)
    label = [i]*len(data)
    labels.extend(label)
    dataset.extend(data)
    


# In[166]:


dataset = dataset[:5000]


# In[167]:


trainx = [i.flatten() for i in dataset]


# In[168]:


trainx = [np.pad(i,(0,10496-i.shape[0])) for i in trainx]


# In[169]:


trainx = pd.DataFrame(trainx).fillna(0)


# In[170]:


trainx["labels"] = labels[0:5000]


# In[171]:


trainx.to_csv("spectro1val.csv")


# In[ ]:


map_classes = {classes[i]:i for i in range(len(classes))}


# In[ ]:




