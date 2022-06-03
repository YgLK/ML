#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import scikeras
import pandas as pd

import os
import pickle


# In[2]:


# method used for saving object as pickle
def save_object_as_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


# In[3]:


# LOAD THE DATA 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# ### Tensorboard

# In[4]:


root_logdir = os.path.join(os.curdir, "tb_logs")
def get_run_logdir(parameter, value):
    import time
    run_id = str(int(time.time())) + "_" + parameter + "_" + str(value)
    return os.path.join(root_logdir, run_id)


# #### Callbacks

# In[5]:


def get_callbacks_list(parameter, value):
    checkpoint_cb = keras.callbacks.ModelCheckpoint(parameter + "_" +  str(value)  + ".h5")
    run_logdir = get_run_logdir(parameter, value)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, min_delta=1.0)

    callbacks_list=[
        # checkpoint_cb,
        tensorboard_cb,
        early_stopping_cb]
    return callbacks_list


# ### Create build_model method

# In[6]:


def build_model(n_hidden=1, n_neurons=25, learning_rate=10e-5, input_shape=[13], optimizer="sgd", momentum=0):
    model = keras.models.Sequential() 
    model.add(keras.layers.InputLayer(input_shape=input_shape)) 
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    if optimizer == "sgd":
        optimizer_prep = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer_prep = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True, momentum=momentum)
    elif optimizer == "momentum":
        optimizer_prep = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer_prep = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=['mse', 'mae'], optimizer=optimizer_prep, metrics=['mse', 'mae'])
    return model


# ### Learning rate test 

# In[7]:


lr_test = [10**(-6),  10**(-5), 10**(-4)]
lr_values = []
lr_mse = []
lr_mae = []


# In[8]:


# clear TensorFlow session
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# perform tests
for lr_val in lr_test:
    callbacks_list = get_callbacks_list("lr", lr_val)
    model = build_model(input_shape=13, learning_rate=lr_val)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        validation_split=0.1,
        callbacks=callbacks_list,
        verbose=False
    )
    lr_values.append(lr_val)
    print(lr_val)
    mse_val = np.mean(history.history['mse']) 
    lr_mse.append(mse_val)
    print(mse_val)
    mae_val = np.mean(history.history['mae']) 
    lr_mae.append(mae_val)
    print(mae_val)
    print("-------------")
    # y_pred = model.predict(X_test)
    # mse_test = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    # mae_test = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    # print("mse_test:", np.mean(mse_test))
    # print("mse_test:", np.mean(mae_test))


# ### Tensorboard analysis

# In[9]:


# %load_ext tensorboard
# %tensorboard --logdir ./my_logs --port=6006


# ### Hidden layers test 

# In[10]:


hl_test = [0, 1, 2, 3]
hl_values = []
hl_mse = []
hl_mae = []


# In[11]:


# clear TensorFlow session
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# perform tests
for hl_val in hl_test:
    callbacks_list = get_callbacks_list("hl", hl_val)
    model = build_model(input_shape=13, n_hidden=hl_val)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        validation_split=0.1,
        callbacks=callbacks_list,
        verbose=False
    )
    hl_values.append(hl_val)
    print(hl_val)
    mse_val = np.mean(history.history['mse']) 
    hl_mse.append(mse_val)
    print(mse_val)
    mae_val = np.mean(history.history['mae']) 
    hl_mae.append(mae_val)
    print(mae_val)
    print("-------------")
    # y_pred = model.predict(X_test)
    # mse_test = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    # mae_test = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    # print("mse_test:", np.mean(mse_test))
    # print("mse_test:", np.mean(mae_test))


# ### Tensorboard analysis

# In[12]:


# %load_ext tensorboard
# %tensorboard --logdir ./my_logs --port=6006


# ### N_neurons test 

# In[13]:


nn_test = [5, 25, 125]
nn_values = []
nn_mse = []
nn_mae = []


# In[14]:


# clear TensorFlow session
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# perform tests
for nn_val in nn_test:
    callbacks_list = get_callbacks_list("nn", nn_val)
    model = build_model(input_shape=13, n_neurons=nn_val)
    history = model.fit(
        X_train, 
        y_train,
        validation_split=0.1,
        epochs=100, 
        callbacks=callbacks_list,
        verbose=False
    )
    nn_values.append(nn_val)
    print(nn_val)
    mse_val = np.mean(history.history['mse']) 
    nn_mse.append(mse_val)
    print(mse_val)
    mae_val = np.mean(history.history['mae']) 
    nn_mae.append(mae_val)
    print(mae_val)
    print("-------------")
    # y_pred = model.predict(X_test)
    # mse_test = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    # mae_test = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    # print("mse_test:", np.mean(mse_test))
    # print("mse_test:", np.mean(mae_test))


# ### Tensorboard analysis

# In[15]:


# %load_ext tensorboard
# %tensorboard --logdir ./my_logs --port=6006


# ### Optimization algorithm test 

# In[16]:


opt_test = ["sgd", "nesterov", "momentum", "adam"]
opt_values = []
opt_mse = []
opt_mae = []


# In[17]:


# clear TensorFlow session
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# perform tests
for opt_val in opt_test:
    callbacks_list = get_callbacks_list("opt", opt_val)
    model = build_model(input_shape=13, optimizer=opt_val)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        validation_split=0.1,
        callbacks=callbacks_list,
        verbose=False
    )
    opt_values.append(opt_val)
    print(opt_val)
    mse_val = np.mean(history.history['mse']) 
    opt_mse.append(mse_val)
    print(mse_val)
    mae_val = np.mean(history.history['mae']) 
    opt_mae.append(mae_val)
    print(mae_val)
    print("-------------")
    # y_pred = model.predict(X_test)
    # mse_test = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    # mae_test = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    # print("mse_test:", np.mean(mse_test))
    # print("mse_test:", np.mean(mae_test))


# ### Tensorboard analysis

# In[18]:


# %load_ext tensorboard
# %tensorboard --logdir ./my_logs --port=6006


# ### Momentum test 

# In[19]:


mom_test = [0.1, 0.5, 0.9]
mom_values = []
mom_mse = []
mom_mae = []


# In[20]:


# clear TensorFlow session
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# perform tests
for mom_val in mom_test:
    callbacks_list = get_callbacks_list("mom", mom_val)
    model = build_model(input_shape=13, optimizer='momentum', momentum=mom_val)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100,
        validation_split=0.1,
        callbacks=callbacks_list,
        verbose=False
    )
    mom_values.append(mom_val)
    print(mom_val)
    mse_val = np.mean(history.history['mse']) 
    mom_mse.append(mse_val)
    print(mse_val)
    mae_val = np.mean(history.history['mae']) 
    mom_mae.append(mae_val)
    print(mae_val)
    print("-------------")
    # y_pred = model.predict(X_test)
    # mse_test = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
    # mae_test = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
    # print("mse_test:", np.mean(mse_test))
    # print("mse_test:", np.mean(mae_test))


# ### Tensorboard analysis

# In[21]:


# %load_ext tensorboard
# %tensorboard --logdir ./tb_logs --port=6006


# ## Save values in pickle

# In[22]:


# learning rate
lr_data = []
for i in range(len(lr_test)):
    lr_data.append((lr_values[i], lr_mse[i], lr_mae[i]))

print(lr_data)
save_object_as_pickle(lr_data, "lr.pkl")


# In[23]:


# hidden layers 
hl_data = []
for i in range(len(hl_test)):
    hl_data.append((hl_values[i], hl_mse[i], hl_mae[i]))

print(hl_data)
save_object_as_pickle(hl_data, "hl.pkl")


# In[24]:


# n_neurons
nn_data = []
for i in range(len(nn_test)):
    nn_data.append((nn_values[i], nn_mse[i], nn_mae[i]))

print(nn_data)
save_object_as_pickle(nn_data, "nn.pkl")


# In[25]:


# optimization
opt_data = []
for i in range(len(opt_test)):
    opt_data.append((opt_values[i], opt_mse[i], opt_mae[i]))

print(opt_data)
save_object_as_pickle(opt_data, "opt.pkl")


# In[26]:


# momentum
mom_data = []
for i in range(len(mom_test)):
    mom_data.append((mom_values[i], mom_mse[i], mom_mae[i]))

print(mom_data)
save_object_as_pickle(mom_data, "mom.pkl")


# ### TASK 2.2

# In[27]:


# TODO  


# In[28]:


param_distribs = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": [5, 25, 125],
    "model__learning_rate": [10**(-6),  10**(-5), 10**(-4)],
    "model__optimizer": ["sgd", "nesterov", "momentum", "adam"],
    "model__momentum": [0.1, 0.5, 0.9]
}


# In[29]:


import scikeras
from scikeras.wrappers import KerasRegressor
es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[30]:


from sklearn.model_selection import RandomizedSearchCV
rnd_search_cv = RandomizedSearchCV(keras_reg,
                                   param_distribs,
                                   n_iter=30,
                                   cv=3,
                                   verbose=2
                                  )
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)


# In[31]:


print(rnd_search_cv.best_score_, rnd_search_cv.best_params_)


# In[32]:


save_object_as_pickle(rnd_search_cv.best_params_, "rnd_search.pkl")


# ### Check saved data

# In[33]:


# check if pickles' contents are saved correctly

print("lr.pkl\n", pd.read_pickle("lr.pkl"), "\n")
print("hl.pkl\n", pd.read_pickle("hl.pkl"), "\n")
print("nn.pkl\n", pd.read_pickle("nn.pkl"), "\n")
print("opt.pkl\n", pd.read_pickle("opt.pkl"), "\n")
print("mom.pkl\n", pd.read_pickle("mom.pkl"), "\n")
print("rnd_search.pkl\n", pd.read_pickle("rnd_search.pkl"), "\n")

