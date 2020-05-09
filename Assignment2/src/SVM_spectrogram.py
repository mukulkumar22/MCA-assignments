#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support


# In[2]:


s1 = pd.read_csv("spectro1.csv")


# In[4]:


s2 = pd.read_csv("spetcro2.csv")


# In[6]:


s2.shape


# In[7]:


s = pd.concat([s1, s2], ignore_index=True)


# In[14]:


trainy = s['labels']


# In[15]:


del s['labels']


# In[17]:


del s['10495']


# In[24]:


trainx = np.array(s)


# In[31]:


classes = os.listdir('Dataset/validation')


# In[32]:


classes


# In[33]:


map_classes = {classes[i]:i for i in range(len(classes))}


# In[34]:


trainy = [map_classes[i] for i in trainy]


# In[37]:


trainy = np.array(trainy)


# In[41]:


s1 = pd.read_csv("spectro1val.csv")


# In[42]:


s2 = pd.read_csv("spetcro2val.csv")


# In[43]:


s = pd.concat([s1, s2], ignore_index=True)


# In[46]:


testy = s["labels"]


# In[48]:


testy = [map_classes[i] for i in testy]


# In[51]:


testy = np.array(testy)


# In[53]:


del s['labels']


# In[55]:


del s['10495']


# In[59]:


testx = np.array(s)


# In[73]:


trainx = pd.DataFrame(trainx).replace([np.inf, -np.inf], np.nan)


# In[76]:


trainx = trainx.fillna(0)


# In[78]:


testx= pd.DataFrame(testx).replace([np.inf, -np.inf], np.nan)


# In[79]:


testx = testx.fillna(0)


# In[82]:





# In[84]:


trainx = normalize(trainx)
testx = normalize(testx)


# In[87]:


clf = SVC(decision_function_shape='ovo')


# In[88]:


clf.fit(trainx,trainy)


# In[90]:


predicted = clf.predict(testx)


# In[97]:


predictedt = clf.predict(trainx)


# In[99]:


(predictedt==trainy).sum()/len(trainx)


# In[116]:


accuracy = (predicted==testy).sum()/len(predicted)
accuracy = accuracy*100
print("accuracy:",accuracy)


# In[113]:


precision_recall_fscore_support(testy,predicted,average='macro')


# In[ ]:




