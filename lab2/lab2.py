#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

import pickle


# In[2]:


# method used for saving object as pickle
def save_object_as_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# In[3]:


mnist = fetch_openml('mnist_784', version=1)


# In[4]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[5]:


print(type(mnist))


# In[6]:


X, y = mnist["data"], mnist["target"].astype(np.uint8)


# In[7]:


print(X.info())


# In[8]:


print(y.describe())


# In[9]:


y_sorted = y.sort_values()


# In[10]:


print(y_sorted.index)


# In[11]:


print(y_sorted)


# In[12]:


X_sorted = X.reindex(y)


# In[13]:


print(X_sorted)


# In[14]:


X_train, X_test = X_sorted[:56000], X_sorted[56000:]
y_train, y_test = y_sorted[:56000], y_sorted[56000:]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[15]:


print("y_train:", np.unique(y_train))
print("y_test:", np.unique(y_test))


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[17]:


print("y_train: ", np.unique(y_train))
print("y_test: ", np.unique(y_test))


# In[18]:


sgd_clf1 = SGDClassifier()
sgd_clf2 = SGDClassifier()

y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)


# In[19]:


print(np.unique(y_train_0))


# In[20]:


sgd_clf1.fit(X_train, y_train_0)


# In[21]:


train_acc = sgd_clf1.score(X_train, y_train_0)
test_acc = sgd_clf1.score(X_test, y_test_0)

print("train_acc", train_acc)
print("test_acc", test_acc)
acc_list = list([train_acc, test_acc])
print(acc_list)

# save acc_list as pickle
save_object_as_pickle(acc_list, "sgd_acc.pkl")


# In[22]:


y_pred = sgd_clf1.predict(X_test)

cv_score = cross_val_score(sgd_clf1, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(cv_score)

# save cv_score as pickle
save_object_as_pickle(cv_score, "sgd_cva.pkl")


# In[23]:


sgd_clf2.fit(X_train, y_train)
y_pred = sgd_clf2.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# save conf_mat as pickle
save_object_as_pickle(conf_mat, "sgd_cmx.pkl")


# In[24]:


# check if pickles are saved correctly
print("acc_list", pd.read_pickle("sgd_acc.pkl"))
print("cs_score", pd.read_pickle("sgd_cva.pkl"))
print("conf_mat\n", pd.read_pickle("sgd_cmx.pkl"))

