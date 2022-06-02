#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# ### Load the data

# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# ### Scale the data

# In[3]:


X_train = X_train / 255.0 
X_test = X_test / 255.0 
y_train = y_train 
y_test = y_test


# ### Show sample image

# In[4]:


# import matplotlib.pyplot as plt 
# plt.imshow(X_train[142], cmap="binary") 
# plt.axis('off')
# plt.show()


# ### Declare class names 

# In[5]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[int(y_train[142])]


# ### Create Sequential model with dense layers

# In[6]:


model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ]
)


# In[7]:


model.summary()


# In[8]:


tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)


# ### Model compilation

# In[32]:


model.compile(
    optimizer='sgd', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics='accuracy'
)


# ### Tensorboard callback

# In[33]:


import os
root_logdir = os.path.join(os.curdir, "image_logs")

def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# ### Train the neural network

# In[34]:


model.fit(
    X_train,
    y_train,
    epochs=20,
    verbose=1,
    validation_split=0.1,
    callbacks=[tensorboard_cb]
)


# ### Check the results arbitrary

# In[37]:


# image_index = np.random.randint(len(X_test))
# image = np.array([X_test[image_index]])
# confidences = model.predict(image)
# confidence = np.max(confidences[0])
# prediction = np.argmax(confidences[0])
# print("Prediction:", class_names[prediction])
# print("Confidence:", confidence)
# print("Truth:", class_names[y_test[image_index]])
# plt.imshow(image[0], cmap="binary")
# plt.axis('off')
# plt.show()


# ### Open the tensorboard

# In[38]:


# %load_ext tensorboard
# %tensorboard --logdir ./image_logs


# ### Save the model

# In[39]:


model.save('fashion_clf.h5')


# ## Regression

# In[42]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[43]:


housing = fetch_california_housing()


# ### Prepare the data 

# In[66]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[67]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# ### Create and train the model

# In[75]:


reg_model = tf.keras.Sequential(
    layers = [
        tf.keras.layers.Dense(
            30, 
            activation="relu", 
              input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ]
)


# In[89]:


reg_model.compile(
    optimizer='sgd', 
    loss=tf.keras.losses.MeanSquaredError(),
)


# In[90]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# ### Prepare logs catch

# In[96]:


root_logdir = os.path.join(os.curdir, "housing_logs")

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[97]:


reg_model.fit(
    X_train,
    y_train,
    epochs=20,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=[es, tensorboard_cb]
)


# In[98]:


# %load_ext tensorboard
# %tensorboard --logdir ./housing_logs


# ### Save the model

# In[99]:


reg_model.save("reg_housing_1.h5")


# # Model 2 - Regression

# ### Create and train the model

# In[124]:


reg_model2 = tf.keras.Sequential(
    layers = [
        tf.keras.layers.Dense(
            30, 
            activation="relu", 
              input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(
            6, 
            activation="relu", 
              input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ]
)


# In[125]:


reg_model2.compile(
    optimizer='sgd', 
    loss=tf.keras.losses.MeanSquaredError(),
)


# In[126]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# ### Prepare logs catch

# In[127]:


root_logdir = os.path.join(os.curdir, "housing_logs")

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[128]:


reg_model2.fit(
    X_train,
    y_train,
    epochs=30,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=[es, tensorboard_cb]
)


# ### Save the model

# In[129]:


reg_model2.save("reg_housing_2.h5")


# # Model 3 - Regression

# ### Create and train the model

# In[130]:


reg_model3 = tf.keras.Sequential(
    layers = [
        tf.keras.layers.Dense(
            60, 
            activation="relu", 
              input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(
            30, 
            activation="relu", 
              input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(
            10, 
            activation="softmax", 
              input_shape=X_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ]
)


# In[131]:


reg_model3.compile(
    optimizer='sgd', 
    loss=tf.keras.losses.MeanSquaredError(),
)


# In[132]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, verbose=1)


# ### Prepare logs catch

# In[133]:


root_logdir = os.path.join(os.curdir, "housing_logs")

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[134]:


reg_model3.fit(
    X_train,
    y_train,
    epochs=30,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=[es, tensorboard_cb]
)


# ### Save the model

# In[135]:


reg_model3.save("reg_housing_3.h5")


# # Tensorboard

# In[136]:


# %load_ext tensorboard
# %tensorboard --logdir ./housing_logs

