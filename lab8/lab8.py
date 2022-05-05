#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle
import pandas as pd


# In[2]:


# method used for saving object as pickle
def save_object_as_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# ### Load the data

# In[3]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()


# In[4]:


data_breast_cancer['data']


# In[5]:


from sklearn.datasets import load_iris 
data_iris = load_iris()


# In[6]:


data_iris['data']


# ## PCA

# ### Data breast cancer data

# #### Not scaled

# In[7]:


pca = PCA(n_components=0.9)
dbc_p = pca.fit_transform(data_breast_cancer['data'])


# In[8]:


print(pca.explained_variance_ratio_)


# #### Scaled

# In[9]:


std_scaler = StandardScaler()
scaled_dbc = std_scaler.fit_transform(data_breast_cancer['data'])
pca_scaled = PCA(n_components=0.9)
dbc_p_scaled = pca_scaled.fit_transform(scaled_dbc)


# In[10]:


# git jest, 7 kolumn powinno wyjsc
print(pca_scaled.explained_variance_ratio_)
dbc_scaled_expl_var = list(pca_scaled.explained_variance_ratio_)
dbc_scaled_expl_var


# #### Save object as pickle

# In[11]:


dbc_scaled_filename = "pca_bc.pkl"
save_object_as_pickle(dbc_scaled_expl_var, dbc_scaled_filename)


# ### Iris data

# #### Not scaled

# In[12]:


pca2 = PCA(n_components=0.9)
iris_p = pca2.fit_transform(data_iris['data'])


# In[13]:


print(pca2.explained_variance_ratio_)


# #### Scaled

# In[14]:


std_scaler2 = StandardScaler()
scaled_iris = std_scaler2.fit_transform(data_iris['data'])
pca_iris_scaled = PCA(n_components=0.9)
iris_p_scaled = pca_iris_scaled.fit_transform(scaled_iris)


# In[15]:


data_iris['data']


# In[16]:


scaled_iris


# In[17]:


print(pca_iris_scaled.explained_variance_ratio_)
ir_scaled_expl_var = list(pca_iris_scaled.explained_variance_ratio_)
ir_scaled_expl_var


# #### Save object as pickle

# In[18]:


ir_scaled_filename = "pca_ir.pkl"
save_object_as_pickle(ir_scaled_expl_var, ir_scaled_filename)


# ## ex. 4

# #### Breast cancer

# In[19]:


bc_comp = pca_scaled.components_


# In[20]:


bc_comp


# In[21]:


l_bc = []
for row in bc_comp:
    idx = row.argmax()
    l_bc.append(idx)
    
l_bc


# #### Save object as pickle

# In[22]:


idx_bc_filename = "idx_bc.pkl"
save_object_as_pickle(l_bc, idx_bc_filename)


# #### Iris

# In[23]:


iris_comp = pca_iris_scaled.components_


# In[24]:


l_iris = []
for row in iris_comp:
    idx = row.argmax()
    l_iris.append(idx)
    
l_iris


# #### Save object as pickle

# In[25]:


idx_ir_filename = "idx_ir.pkl"
save_object_as_pickle(l_iris, idx_ir_filename)


# ### Check saved Pickles contents

# In[26]:


# check if pickles' contents are saved correctly

print("pca_bc.pkl\n", pd.read_pickle("pca_bc.pkl"), "\n")
print("pca_ir.pkl\n", pd.read_pickle("pca_ir.pkl"), "\n")
print("idx_bc.pkl\n", pd.read_pickle("idx_bc.pkl"), "\n")
print("idx_ir.pkl\n", pd.read_pickle("idx_ir.pkl"))

