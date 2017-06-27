# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:14:03 2017

@author: HSIN
"""

from keras.engine import Layer
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.merge import Add,Dot,Concatenate,Multiply
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam,SGD
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import time
import xgboost as xgb
import keras

stime = time.time()

train = pd.read_csv('./train_lon_lat_missing.csv', index_col=0)
test = pd.read_csv('./test_lon_lat_missing.csv', index_col=0)

# to predict lon
lon_train = pd.read_csv('./lon_train.csv',index_col=0)
lon_test = pd.read_csv('./lon_test.csv',index_col=0)
lon_target = pd.read_csv('./lon_target.csv',index_col=0, header=None)

# to predict lat
lat_train = pd.read_csv('./lat_train.csv', index_col=0)
lat_test = pd.read_csv('./lat_test.csv', index_col=0)
lat_target = pd.read_csv('./lat_target.csv', index_col=0, header=None)


def get_nn_model():
    user_input = Input(shape=[6])
    hidden = Dense(12,activation='relu')(user_input)
    hidden = Dense(36,activation='relu')(hidden)
    hidden = Dense(36,activation='relu')(hidden)
    hidden = Dense(18,activation='relu')(hidden)
    output = Dense(12,activation='relu')(hidden)
    output = Dense(4,activation='relu')(hidden)
    output = Dense(1)(hidden)
    model = Model(user_input,output)
    adam = Adam(lr=0.06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model
'''
# lon xgb
lon_est = xgb.XGBRegressor(
                            max_depth = 15,
                            learning_rate = 0.05,
                            n_estimators = 400,
                            gamma = 0.03,
                            min_child_weight = 1.5,
                            subsample = 0.9,
                            colsample_bytree = 0.9
                            )
                        
                        
                        
lon_est.fit(lon_train, lon_target)
lon_pred = lon_est.predict(lon_test)


# lat xgb
lat_est = xgb.XGBRegressor(
                            max_depth = 15,
                            learning_rate = 0.05,
                            n_estimators = 300,
                            gamma = 0.03,
                            min_child_weight = 0.5,
                            subsample = 0.9,
                            colsample_bytree = 0.9
                            )
                            
lat_est.fit(lat_train, lat_target)
lat_pred = lat_est.predict(lat_test) 
'''
epoch_nb = 10000

earlystopping = EarlyStopping(monitor='loss', patience = 100., verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='tmp.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='loss',
                             mode='min')

lon_model = get_nn_model()
lon_model.fit(lon_train.as_matrix(), lon_target.as_matrix(),batch_size=5000,
              epochs=epoch_nb,callbacks=[earlystopping,checkpoint])
lon_model.load_weights('tmp.hdf5')
lon_pred = lon_model.predict(lon_test.as_matrix()) 

earlystopping = EarlyStopping(monitor='loss', patience = 100., verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='tmp.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='loss',
                             mode='min')

lat_model = get_nn_model()
lat_model.fit(lat_train.as_matrix(), lat_target.as_matrix(),batch_size=5000,
              epochs=epoch_nb,callbacks=[earlystopping,checkpoint])
lat_model.load_weights('tmp.hdf5')
lat_pred = lat_model.predict(lat_test.as_matrix()) 

lon_model.save('lon_nn_model.h5')
lat_model.save('lat_nn_model.h5')


# get train test index
train_ind = lon_test.index[lon_test.index < train.shape[0]]
test_ind = lon_test.index[lon_test.index >= train.shape[0]]


# replace train lon lat by predicted value
train.ix[train_ind,'longitude'] = lon_pred[0:len(train_ind)] 
train.ix[train_ind,'latitude' ]= lat_pred[0:len(train_ind)]

# replace test lon lat by predicted value
test.ix[test_ind,'longitude'] = lon_pred[len(train_ind):lon_pred.shape[0]] 
test.ix[test_ind,'latitude'] = lat_pred[len(train_ind):lat_pred.shape[0]]

      

train.to_csv('train_lon_lat_nn_predicted.csv')
test.to_csv('test_lon_lat_nn_predicted.csv')

etime = float(time.time()-stime)
print('Time:', etime)