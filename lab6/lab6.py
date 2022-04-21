#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import pickle


# In[2]:


# method used for saving object as pickle
def save_object_as_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# ## Import the data 

# In[3]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True) 
print(data_breast_cancer['DESCR'])


# In[4]:


# split the dataset
X = data_breast_cancer['data'][['mean texture', 'mean symmetry']]
y = data_breast_cancer['frame'].target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# 3 bazowe + voting hard + voting soft - tj. razem 5 elementow 


# In[6]:


X_train.head()


# In[7]:


y_train.head()


# # Ensemble

# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


# In[9]:


dtr_clf = DecisionTreeClassifier(random_state=42)
log_clf = LogisticRegression(random_state=42)
knn_clf = KNeighborsClassifier()

voting_clf_hard = VotingClassifier(
    estimators=[('lr', log_clf),
                ('dtr', dtr_clf),
                ('knn', knn_clf)],
    voting='hard')

voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_clf),
                ('dtr', dtr_clf),
                ('knn', knn_clf)],
    voting='soft')


# ### Calculate accuracy

# In[10]:


acc_list = []

for clf in (dtr_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print("TRAIN", clf.__class__.__name__,
      train_acc)
    print("TEST", clf.__class__.__name__,
          test_acc)
    acc_list.append((train_acc, test_acc))
    
# WARNING! First voting classifier has 'hard' type of voting
# second voting classifier has 'soft' type of voting (it's not the same classifier!)


# In[11]:


# test if data is properly saved
for tup in acc_list:
    print(tup)


# ### Save data in the pickles

# In[12]:


save_object_as_pickle(acc_list, "acc_vote.pkl")

clf_list = [dtr_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft]

save_object_as_pickle(clf_list, "vote.pkl")


# ### Check if data saved properly

# In[13]:


print("acc_vote.pkl\n", pd.read_pickle("acc_vote.pkl"), "\n")
print("vote.pkl\n", pd.read_pickle("vote.pkl"), "\n")


# ## 30 decision trees (ex. 5)
# * Bagging,
# * Bagging z wykorzystaniem 50% instancji,
# * Pasting,
# * Pasting z wykorzystaniem 50% instancji, oraz 
# * Random Forest,
# * AdaBoost,
# * Gradient Boosting.
# 

# In[14]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[15]:


# Bagging
bgn_clf_full = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=True, random_state=42)
bgn_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=True, max_samples=0.5, random_state=42)
# Pasting
pst_clf_full = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, random_state=42)
pst_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, max_samples=0.5, random_state=42)
# Random forest
rnd_clf = RandomForestClassifier(n_estimators=30, random_state=42)
# Ada boost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=30, random_state=42)
# Gradient boosting
gdb_clf = GradientBoostingClassifier(n_estimators=30, random_state=42)


# In[16]:


classificators = [bgn_clf_full, bgn_clf_half, pst_clf_full, pst_clf_half, rnd_clf, ada_clf, gdb_clf]

clf_acc = []

for clf in classificators:
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print("TRAIN", clf.__class__.__name__,
      train_acc)
    print("TEST", clf.__class__.__name__,
          test_acc, "\n")
    clf_acc.append((train_acc, test_acc))
    
# Bagging oraz Random forest daje takie same wyniki, dziwne - sprawdzic


# ### Save data in the pickles

# In[17]:


save_object_as_pickle(clf_acc, "acc_bag.pkl")

save_object_as_pickle(classificators, "bag.pkl")


# ### Check if data saved properly

# In[18]:


print("acc_bag.pkl\n", pd.read_pickle("acc_bag.pkl"), "\n")
print("bag.pkl\n", pd.read_pickle("bag.pkl"), "\n")


# ## Sampling

# In[19]:


# Odnosnie zad 7: Tutaj BaggingClassifier z decision tree i max_features=2, 
# reszta ustawien domyslna


# In[20]:


# prepare the data
X_all = data_breast_cancer['data']
y_all = data_breast_cancer['frame'].target
X_train, X_test, y_train, y_test= train_test_split(X_all, y_all, test_size=0.2, random_state=42)


# In[21]:


X_train.head()


# In[22]:


y_train.head()


# In[23]:


smp_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, 
                            max_samples=0.5, max_features=2, 
                            bootstrap=True, bootstrap_features=False,
                            random_state=42)


# In[24]:


smp_clf.fit(X_train, y_train)
y_pred_test = smp_clf.predict(X_test)
y_pred_train = smp_clf.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print("TRAIN", smp_clf.__class__.__name__,
  train_acc)
print("TEST", smp_clf.__class__.__name__,
      test_acc, "\n")

smp_acc = [train_acc, test_acc]


# ### Save data in the pickles

# In[25]:


save_object_as_pickle(smp_acc, "acc_fea.pkl")

save_object_as_pickle(smp_clf, "fea.pkl")


# ### Check if data saved properly

# In[26]:


print("acc_fea.pkl\n", pd.read_pickle("acc_fea.pkl"), "\n")
print("fea.pkl\n", pd.read_pickle("fea.pkl"), "\n")


# ## Classifier ranking

# In[27]:


smp_clf.estimators_


# In[28]:


smp_clf.estimators_features_


# In[29]:


# list of (train_accuracy, test_accuracy) elements
rank_acc = []
# list of [first_feautre_name, second_feauter_name] elements
rank_fea = []

for clf, fea_pair in zip(smp_clf.estimators_, smp_clf.estimators_features_):
#     clf.fit(X_train, y_train)
    first_fea = X_all.columns[fea_pair[0]]
    second_fea = X_all.columns[fea_pair[1]]
    columns = [first_fea, second_fea]
    y_pred_test = clf.predict(X_test[columns])
    y_pred_train = clf.predict(X_train[columns])
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print("TRAIN", clf.__class__.__name__,
      train_acc)
    print("TEST", clf.__class__.__name__,
          test_acc, "")
    rank_acc.append((train_acc, test_acc))
    print("Features: ", first_fea, " ", second_fea, "\n")
    rank_fea.append([first_fea, second_fea])


# In[30]:


# prepare dataframe columns
train_acc= [x[0] for x in rank_acc] 
test_acc= [x[1] for x in rank_acc]
ranking = pd.DataFrame({
    "train accuracy": train_acc,
    "test accuracy": test_acc,
    "features": rank_fea
})


# In[31]:


ranking = ranking.sort_values(by=['test accuracy', 'train accuracy'], ascending=False)


# In[32]:


ranking


# ### Save data in the pickles

# In[33]:


save_object_as_pickle(ranking, "acc_fea_rank.pkl")


# ## Check if data saved properly

# In[34]:


print("acc_fea_rank.pkl\n", pd.read_pickle("acc_fea_rank.pkl"), "\n")


# # Summary

# In[35]:


print("acc_fea_rank.pkl\n", pd.read_pickle("acc_fea_rank.pkl"), "\n")
print("acc_vote.pkl\n", pd.read_pickle("acc_vote.pkl"), "\n")
print("vote.pkl\n", pd.read_pickle("vote.pkl"), "\n")
print("acc_bag.pkl\n", pd.read_pickle("acc_bag.pkl"), "\n")
print("bag.pkl\n", pd.read_pickle("bag.pkl"), "\n")
print("acc_fea.pkl\n", pd.read_pickle("acc_fea.pkl"), "\n")
print("fea.pkl\n", pd.read_pickle("fea.pkl"), "\n")
print("acc_fea_rank.pkl\n", pd.read_pickle("acc_fea_rank.pkl"), "\n")

