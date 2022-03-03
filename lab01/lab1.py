#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import os
import tarfile
import urllib.request

HOUSING_PATH = os.path.join("data")
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
 os.makedirs(housing_path, exist_ok=True)
 tgz_path = os.path.join(housing_path, "housing.tgz")
 urllib.request.urlretrieve(housing_url, tgz_path)
 housing_tgz = tarfile.open(tgz_path)
 housing_tgz.extractall(path=housing_path)
 housing_tgz.close()


# In[3]:


fetch_housing_data()


# In[4]:


df = pd.read_csv("data/housing.csv")

print(df.head())


# In[5]:


print(df.info())


# In[6]:


print(df['ocean_proximity'].dtype)


# In[7]:


print(df['ocean_proximity'].value_counts())


# In[8]:


print(df['ocean_proximity'].describe())


# In[9]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[10]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[11]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[12]:


df.corr()["median_house_value"]    .sort_values(ascending=False)    .reset_index()     .rename(columns={'index': 'atrybut', 'median_house_value': 'wspolczynnik_korelacji'})    .to_csv("korelacja.csv", index=False)


# In[13]:


import seaborn as sns

sns.pairplot(df)


# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
len(train_set),len(test_set)


# In[15]:


print(train_set)


# In[16]:


print(test_set)


# In[17]:


print(train_set.corr())


# In[18]:


print(test_set.corr())


# In[19]:


train_set.to_pickle("train_set.pkl")
test_set.to_pickle("test_set.pkl")


# In[20]:


# Check if pickles are read properly
# pd.read_pickle('train_set.pkl').head()


# In[21]:


# pd.read_pickle('test_set.pkl').head()

