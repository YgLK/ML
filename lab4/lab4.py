#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

import pickle


# # Breast cancer dataset

# In[2]:


data = datasets.load_breast_cancer(as_frame=True)
print(data['DESCR'])


# In[3]:


data['data']


# In[4]:


# split the dataset
X = data['data']
y = data['frame'].target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)


# ### SVM Imports

# In[5]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# ## SVM - area and smoothness

# In[6]:


X_train_area_smooth = X_train[['mean area', 'mean smoothness']]
X_test_area_smooth = X_test[['mean area', 'mean smoothness']]


# ### Without scaling

# In[7]:


svm_clf_without_scaling = Pipeline([("linear_svc", LinearSVC(C=1, loss="hinge"))])


# In[8]:


# train the model
svm_clf_without_scaling.fit(X_train_area_smooth, y_train)


# In[9]:


y_test_no_scaling_pred = svm_clf_without_scaling.predict(X_test_area_smooth)
y_train_no_scaling_pred = svm_clf_without_scaling.predict(X_train_area_smooth)


# In[10]:


test_no_scaling_acc = accuracy_score(y_test, y_test_no_scaling_pred)
train_no_scaling_acc = accuracy_score(y_train, y_train_no_scaling_pred)
print('test_no_scaling_acc', test_no_scaling_acc)
print('train_no_scaling_acc', train_no_scaling_acc)


# ### With scaling

# In[11]:


svm_clf_with_scaling = Pipeline([("scaler", StandardScaler()),
                                    ("linear_svc", LinearSVC(C=1,
                                                             loss="hinge"))])


# In[12]:


# train the model
svm_clf_with_scaling.fit(X_train_area_smooth, y_train)


# In[13]:


y_test_scaling_pred = svm_clf_with_scaling.predict(X_test_area_smooth)
y_train_scaling_pred = svm_clf_with_scaling.predict(X_train_area_smooth)


# In[14]:


test_scaling_acc = accuracy_score(y_test, y_test_scaling_pred)
train_scaling_acc = accuracy_score(y_train, y_train_scaling_pred)
print('test_scaling_acc', test_scaling_acc)
print('train_scaling_acc', train_scaling_acc)


# ### Save the data as pickle

# In[15]:


bc_acc_res = [train_no_scaling_acc, test_no_scaling_acc, train_scaling_acc, test_scaling_acc]
# save accuracy results as pickle
filename = "bc_acc.pkl"
with open(filename, 'wb') as file:
        pickle.dump(bc_acc_res, file, pickle.HIGHEST_PROTOCOL)


# # Iris dataset

# In[16]:


data2 = datasets.load_iris(as_frame=True)
print(data2['DESCR'])


# In[17]:


# split the dataset
X2 = data2['data']
y2 = data2['frame'].target
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)


# ### SVM for sepal length and width - gatunek VIRGINICA (dla tego jednego gatunku predykcje robimy) !!!

# In[18]:


X2


# In[19]:


y2


# In[20]:


X2_train_len_wid = X2_train[['petal length (cm)', 'petal width (cm)']]
X2_test_len_wid = X2_test[['petal length (cm)', 'petal width (cm)']]


# #### Without scaling

# In[21]:


svm_clf2_without_scaling = Pipeline([("linear_svc", LinearSVC(C=1, loss="hinge"))])


# In[22]:


# train the model
svm_clf2_without_scaling.fit(X2_train_len_wid, y2_train)


# In[23]:


y_test2_no_scaling_pred = svm_clf2_without_scaling.predict(X2_test_len_wid)
y_train2_no_scaling_pred = svm_clf2_without_scaling.predict(X2_train_len_wid)


# In[24]:


test2_no_scaling_acc = accuracy_score(y2_test, y_test2_no_scaling_pred)
train2_no_scaling_acc = accuracy_score(y2_train, y_train2_no_scaling_pred)
print('test2_no_scaling_acc', test2_no_scaling_acc)
print('y_train2_no_scaling_pred', train2_no_scaling_acc)


# ### With scaling

# In[25]:


svm_clf2_with_scaling = Pipeline([("scaler", StandardScaler()),
                                    ("linear_svc", LinearSVC(C=1,
                                                             loss="hinge"))])


# In[26]:


# train the model
svm_clf2_with_scaling.fit(X2_train_len_wid, y2_train)


# In[27]:


y2_test_scaling_pred = svm_clf2_with_scaling.predict(X2_test_len_wid)
y2_train_scaling_pred = svm_clf2_with_scaling.predict(X2_train_len_wid)


# In[28]:


test2_scaling_acc = accuracy_score(y2_test, y2_test_scaling_pred)
train2_scaling_acc = accuracy_score(y2_train, y2_train_scaling_pred)
print('test2_scaling_acc', test2_scaling_acc)
print('train2_scaling_acc', train2_scaling_acc)


# ### Save data as pickle

# In[29]:


iris_acc_res = [train2_no_scaling_acc, test2_no_scaling_acc, train2_scaling_acc, test2_scaling_acc]
# save accuracy results as pickle
filename2 = "iris_acc.pkl"
with open(filename2, 'wb') as file:
        pickle.dump(iris_acc_res, file, pickle.HIGHEST_PROTOCOL)


# # Check saved Pickles contents
# 

# In[30]:


# check if pickles' contents are saved correctly
print("bc_acc_res\n", pd.read_pickle("bc_acc.pkl"), "\n")
print("iris_acc_res\n", pd.read_pickle("iris_acc.pkl"))

