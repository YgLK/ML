#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.datasets import fetch_openml
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

import pickle
# measure execution time
import timeit


# In[11]:


# method used for saving object as pickle
def save_object_as_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# ### Load the data

# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# ### Clustering

# In[6]:


cluster_nums = [8,9,10,11,12]
silh_scores = []

# Mesaure execution time
# starttime = timeit.default_timer()
# print("The start time is :",starttime)


for k in cluster_nums:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    s_score = silhouette_score(X, kmeans.labels_)
    silh_scores.append(s_score)
    

print("Silhouette score list:", silh_scores)

# get execution time
# print(f"The time difference is : {0} seconds").format(timeit.default_timer() - starttime)


# In[57]:


plt.plot(cluster_nums, silh_scores) 


# ### Save silhouette score list in pickle

# In[59]:


kmeans_sil_filename = "kmeans_sil.pkl"

save_object_as_pickle(silh_scores, kmeans_sil_filename)


# ### Confusion matrix for 10 clusters Kmeans

# In[33]:


kmeans10 = KMeans(n_clusters=10, random_state=42)
y_pred10 = kmeans10.fit_predict(X)

conf_matrix10 = confusion_matrix(y, y_pred10)


# In[34]:


conf_matrix10 


# #### 3.5 Find max value idx of each row

# In[ ]:


max_val_idxs = []
for row in conf_matrix10:
    idx = np.argmax(row)
    max_val_idxs.append(idx)

# without duplicates 
no_dupl_max_vals = set(max_val_idxs)
# LIST should be saved (I forgot to do that...)
to_be_saved = list(no_dupl_max_vals)


# In[61]:


no_dupl_max_vals


# In[ ]:


# this should be saved into kmeans_argmax.pkl file
to_be_saved


# Number for each row in matrix: <br>
# 9 1 4 0 3(5) 0(3/8) 6(2) 5(3) 8 3(5)

# ### Save numbers in pickle

# In[66]:


no_dupl_max_vals_filename = "kmeans_argmax.pkl"

# save_object_as_pickle(no_dupl_max_vals, no_dupl_max_vals_filename)
# correct way (list instead of set should be )
save_object_as_pickle(to_be_saved, no_dupl_max_vals_filename)


# ### DBSCAN

# In[5]:


norm_val = []

for i in range(300):
    for j in range(len(X)):
        if i == j:
            continue
        norm_val.append(np.linalg.norm(X[i] - X[j]))


# In[18]:


# norm_val_sorted = np.unique(np.sort(norm_val))
# dist = norm_val_sorted[:10]
dist = np.sort(norm_val)[:10]
print(dist)


# ### Save distance list in pickle

# In[19]:


dist_filename = "dist.pkl"

save_object_as_pickle(dist, dist_filename)


# ### Calculate mean distance

# In[20]:


s = dist[:3].mean()
s
# jest git, 526 wyszlo tez u innych


# In[21]:


print("start ", s)
print("end ", s+0.1*s)
print("step ", 0.04*s)
s_values = np.arange(s, s+0.1*s, step=0.04*s)


# In[24]:


s_values


# In[36]:


dbscan = DBSCAN(eps=315.8)
dbscan.fit(X)
unique = set(dbscan.labels_)
print(unique)


# In[22]:


unique_label_num = []

for epsilon in s_values:
    dbscan = DBSCAN(eps=epsilon)
    dbscan.fit(X)
    unique = set(dbscan.labels_)
    print(unique)
    unique_label_num.append(len(unique))


# In[23]:


unique_label_num


# ### Save numbers in pickle

# In[33]:


uniq_label_name = "dbscan_len.pkl"

save_object_as_pickle(unique_label_num, uniq_label_name)


# ## Check saved Pickles contents

# In[34]:


# check if pickles' contents are saved correctly

print("kmeans_sil\n", pd.read_pickle("kmeans_sil.pkl"), "\n")
print("kmeans_argmax\n", pd.read_pickle("kmeans_argmax.pkl"), "\n")
print("dist\n", pd.read_pickle("dist.pkl"), "\n")
print("dbscan_len\n", pd.read_pickle("dbscan_len.pkl"))

