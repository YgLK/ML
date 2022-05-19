#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import load_iris 
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


# method used for saving object as pickle
def save_object_as_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# ### Load the data

# In[ ]:


iris = load_iris(as_frame=True)


# In[ ]:


pd.concat([iris.data, iris.target], axis=1).plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    colormap='viridis'
)


# In[ ]:


iris.data


# In[ ]:


X = iris.data[['petal length (cm)','petal width (cm)']]
y = iris.target


# In[ ]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)


# In[ ]:


y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)


# #### for 0 target value

# In[ ]:


# for 0 target value
per_clf_0 = Perceptron()
per_clf_0.fit(X_train, y_train_0)


# In[ ]:


y_pred_train_0 = per_clf_0.predict(X_train)
y_pred_test_0 = per_clf_0.predict(X_test)


# In[ ]:


acc_train_0 = accuracy_score(y_train_0, y_pred_train_0)
acc_test_0 = accuracy_score(y_test_0, y_pred_test_0)
print("acc_train_0", acc_train_0)
print("acc_test_0", acc_test_0)


# #### for 1 target value

# In[ ]:


y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)


# In[ ]:


# for 1 target value
per_clf_1 = Perceptron()
per_clf_1.fit(X_train, y_train_1)


# In[ ]:


y_pred_train_1 = per_clf_1.predict(X_train)
y_pred_test_1 = per_clf_1.predict(X_test)


# In[ ]:


acc_train_1 = accuracy_score(y_train_1, y_pred_train_1)
acc_test_1 = accuracy_score(y_test_1, y_pred_test_1)
print("acc_train_1", acc_train_1)
print("acc_test_1", acc_test_1)


# #### for 2 target value

# In[ ]:


y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)


# In[ ]:


# for 2 target value
per_clf_2 = Perceptron()
per_clf_2.fit(X_train, y_train_2)


# In[ ]:


y_pred_train_2 = per_clf_2.predict(X_train)
y_pred_test_2 = per_clf_2.predict(X_test)


# In[ ]:


acc_train_2 = accuracy_score(y_train_2, y_pred_train_2)
acc_test_2 = accuracy_score(y_test_2, y_pred_test_2)
print("acc_train_2", acc_train_2)
print("acc_test_2", acc_test_2)


# #### weights

# In[ ]:


print("0:  bias weight", per_clf_0.intercept_)
print("Input weights (w1, w2): ", per_clf_0.coef_)


# In[ ]:


print("1:  bias weight", per_clf_1.intercept_)
print("Input weights (w1, w2): ", per_clf_1.coef_)


# In[ ]:


print("0:  bias weight", per_clf_2.intercept_)
print("Input weights (w1, w2): ", per_clf_2.coef_)


# #### Save accuracy lists and weight tuple in the pickles

# In[ ]:


# accuracy
per_acc = [(acc_train_0, acc_test_0), (acc_train_1, acc_test_1), (acc_train_2, acc_test_2)]
filename = "per_acc.pkl"
save_object_as_pickle(per_acc, filename)
print("per_acc\n", per_acc)


# In[ ]:


# weights
per_wght = [(per_clf_0.intercept_[0], per_clf_0.coef_[0][0], per_clf_0.coef_[0][1]), (per_clf_1.intercept_[0], per_clf_1.coef_[0][0], per_clf_1.coef_[0][1]), (per_clf_2.intercept_[0], per_clf_2.coef_[0][0], per_clf_2.coef_[0][1])]
filename = "per_wght.pkl"
save_object_as_pickle(per_wght, filename)
print("per_wght\n", per_wght)


# ### Perceptron, XOR

# In[ ]:


X = np.array([[0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])
y = np.array([0,
   1, 
   1, 
   0])


# In[ ]:


per_clf_xor = Perceptron()
per_clf_xor.fit(X, y)


# In[ ]:


pred_xor = per_clf_xor.predict(X)
xor_acc = accuracy_score(y, pred_xor)

print("xor_accuracy:", xor_acc)


# In[ ]:


print("XOR:  bias weight", per_clf_xor.intercept_)
print("Input weights (w1, w2): ", per_clf_xor.coef_)


# ### 2nd Perceprton, XOR

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


while True:
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(2, activation="relu", input_dim=2))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.085), 
                  metrics=["binary_accuracy"])

    history = model.fit(X, y, epochs=100, verbose=False)
    predict_prob=model.predict(X)
    print(predict_prob)
    print(history.history['binary_accuracy'][-1])
    if predict_prob[0] < 0.1 and predict_prob[1] > 0.9 and predict_prob[2] > 0.9 and predict_prob[3] < 0.1:
        weights = model.get_weights()
        break


# ### Save data to pickle

# In[ ]:


print("weights\n", weights)


# In[ ]:


filename = "mlp_xor_weights.pkl"
save_object_as_pickle(weights, filename)


# ### Check saved Pickles contents

# In[ ]:


# check if pickles' contents are saved correctly

print("per_acc.pkl\n", pd.read_pickle("per_acc.pkl"), "\n")
print("per_wght.pkl\n", pd.read_pickle("per_wght.pkl"), "\n")
print("mlp_xor_weights.pkl\n", pd.read_pickle("mlp_xor_weights.pkl"), "\n")

