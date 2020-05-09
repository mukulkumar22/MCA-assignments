#!/usr/bin/env python
# coding: utf-8

# In[259]:


from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support
from scipy.io import wavfile
from scipy.fftpack import dct
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import librosa.display
import math
import os
import warnings
warnings.filterwarnings("ignore")


# ## MFCC Code

# In[2]:


def mel_2_hz(mel):
    return  700*(10**(mel/2595)-1)
    


# In[3]:


def hz_2_mel(f):
    return 2595 * math.log10(1+(f/700))
    


# In[4]:


def closest_fftbin(frequency,resolution,sample_rate):
    return math.floor((resolution+1)*frequency/sample_rate)


# In[5]:


def get_coefficients(lbin,pbin,hbin,lcoef = 0.1,pcoef = 1):
    coeffs = []
    stepsize = (pcoef-lcoef)/(pbin-lbin)
    
    for i in range((pbin-lbin)):
        coeffs.append(lcoef+i*stepsize)
    coeffs.append(pcoef)
    
    stepsize = (pcoef-lcoef)/(hbin-pbin)
    for i in range((hbin-pbin),0,-1):
        coeffs.append(lcoef+(stepsize*i)-stepsize)
    return coeffs
    


# In[6]:


def create_frames(data,frame_size,overlap,apply_hamming = True):
    frames = []
    modified = []
    startinds = [0 +(frame_size-overlap)*i for i in range(len(data))]
    startinds = [i for i in startinds if i<=len(data)]
    for i in startinds:
        frames.append(data[i:i+frame_size])
    for i in frames:
        if len(i)!=0 and len(i)<frame_size:
            i = np.append(i,[0]*(frame_size-len(i)))
        modified.append(i)
        
    frames = np.array(modified)
    
    hamming_frames = []
    mult = np.hamming(frame_size)
    if apply_hamming:
        for i in frames:
            hamming_frames.append(np.multiply(mult,i))
        return np.array(hamming_frames)
    
    else:
        return frames
    
    


# In[7]:


def get_periodograms(frames,resolution):
    result = []
    
    for i in frames:
        res = np.fft.fft(i,resolution)
        resu = np.absolute(res[0:len(res)//2+1])
        res = np.square((1/len(res))*resu)
        result.append(res)
        
    return np.array(result)


# In[8]:


def get_filters(sample_rate,lower_frequency=0,upper_frequency = 8000,resolution=512,n_filters = 26):
    l = hz_2_mel(lower_frequency)
    h = hz_2_mel(upper_frequency)
    k = np.linspace(l,h,n_filters+2)
    f = [mel_2_hz(i) for i in k]
    j = [closest_fftbin(i,resolution,sample_rate) for i in f]
    step1 = (h-l)/(n_filters+1)
    mel_points = [(l+(i*step1)) for i in range(0,n_filters+2)]
    hzpoints = [mel_2_hz(i) for i in mel_points]
    bins = [closest_fftbin(i,sample_rate=sample_rate,resolution=resolution) for i in hzpoints]
    
    
    filters = []
    for i in range(len(bins)-2):
        filterr = []
        lb = bins[i]
        pb = bins[i+1]
        hb = bins[i+2]
        lc = 0.1
        pc = 1
        
        step1 = (pc-lc)/(pb-lb)
        for ii in range(pb-lb):
            filterr.append(lc +(step1*ii))
        filterr.append(pc)
        
        step2 = (pc-lc)/(hb-pb)
        for ii in range(hb-pb,0,-1):
            filterr.append(lc+((step2*ii)-step2))
            
        filters.append((filterr,[lb,pb,hb]))
        
    filter_bank = np.zeros((n_filters,resolution//2+1))
    for i in range(n_filters):
        l = filters[i][1][0]
        s = len(filters[i][0])
        filter_bank[i,l:l+s] = filters[i][0]
        
    return filter_bank
        
    


# In[9]:


def apply_filters(frames,filters):
    return np.dot(frames,filters.T)


# In[10]:


def get_mfcc_coeff(filtered_data):
    logenergy =  np.log10(filtered_data)
    dctt = dct(logenergy)
    mfcc = dctt[:,1:13]
    return mfcc
    


# In[11]:


def plot_mfcc(mfcc,sample_rate):
    librosa.display.specshow(mfcc,sr = sample_rate,x_axis = "time")


# In[335]:


def mfcc(file,resolution,frame_size,overlap,plot = False):
    sample_rate,data = wavfile.read(file)
    frames = create_frames(data,frame_size=frame_size,overlap=overlap)
    frames = get_periodograms(frames,resolution)
    filters = get_filters(sample_rate)
    result = apply_filters(frames,filters)
    mfcc = get_mfcc_coeff(result)
    
    if plot:
        plot_mfcc(mfcc)
        
    return mfcc  


# In[326]:


def mfcc_withnoise(file,resolution,frame_size,overlap,plot = False,noise = "dude_miaowing.wav"):
    path = 'Dataset/_background_noise_/'+noise
    
    sample_rate,data = wavfile.read(file)
    nr,nd = wavfile.read(path,sample_rate)
    data = data+nd[0:len(data)]
    
    frames = create_frames(data,frame_size=frame_size,overlap=overlap)
    frames = get_periodograms(frames,resolution)
    filters = get_filters(sample_rate)
    result = apply_filters(frames,filters)
    mfcc = get_mfcc_coeff(result)
    
    if plot:
        plot_mfcc(mfcc)
        
    return mfcc  


# In[336]:


def create_dataset(sound_class,path = "train"):
    if path == 'train':
        path = path_training
    else:
        path = path_validation
    dataset = []
    files = os.listdir(path+sound_class)
    for i in files:
        try:
            
            dataset.append(mfcc(path+sound_class+"/"+i,512,256,64))
            #dataset.append(mfcc_withnoise(path+sound_class+"/"+i,512,400,160))
        except ValueError:
            continue
    
    return dataset


# ## SVM for classification

# In[13]:


path_training = 'Dataset/training/'
path_validation = 'Dataset/validation/'


# In[14]:


classes = os.listdir('Dataset/validation')


# In[ ]:





# In[337]:


dataset = []
labels = []
for i in classes:  
    data = create_dataset(i)
    label = [i]*len(data)
    labels.extend(label)
    dataset.extend(data)
    
trainx = [i.flatten() for i in dataset]
thres = trainx[0].shape[0]
trainx = [np.pad(i,(0,thres-i.shape[0])) for i in trainx]

map_classes = {classes[i]:i for i in range(len(classes))}
trainy = [map_classes[i] for i in labels]

trainx = pd.DataFrame(trainx).fillna(0)
trainx = normalize(trainx)
trainy = np.array(trainy)


# In[260]:


dataset = []
labels = []
for i in classes:  
    data = create_dataset(i,"valid")
    label = [i]*len(data)
    labels.extend(label)
    dataset.extend(data)
    
testx = [i.flatten() for i in dataset]
testx = [np.pad(i,(0,thres-i.shape[0])) for i in testx]
testx = pd.DataFrame(testx).fillna(0)
testx = normalize(testx)
testy = [map_classes[i] for i in labels]
testy = np.array(testy)


# In[338]:


clf = SVC(decision_function_shape='ovo')
clf.fit(trainx,trainy)


# In[339]:


predicted = clf.predict(testx)
accuracy = (predicted==testy).sum()/len(predicted)
p = precision_recall_fscore_support(testy,predicted,average='macro')


# In[340]:


precision = p[0]
recall = p[1]
F1_score = p[2]


# In[341]:


print("accuracy:",str(round(accuracy,4)*100)+"%")
print("precision:",round(precision,4))
print("recall",round(recall,4))
print("F1 Score:",round(F1_score,4))

