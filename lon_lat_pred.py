# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:14:03 2017

@author: HSIN
"""

import pandas as pd
import time
import xgboost as xgb

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


# get train test index
train_ind = lon_test.index[lon_test.index < train.shape[0]]
test_ind = lon_test.index[lon_test.index >= train.shape[0]]


# replace train lon lat by predicted value
train.ix[train_ind,'longitude'] = lon_pred[0:len(train_ind)] 
train.ix[train_ind,'latitude' ]= lat_pred[0:len(train_ind)]

# replace test lon lat by predicted value
test.ix[test_ind,'longitude'] = lon_pred[len(train_ind):lon_pred.shape[0]] 
test.ix[test_ind,'latitude'] = lat_pred[len(train_ind):lat_pred.shape[0]]

      

train.to_csv('train_lon_lat_predicted.csv')
test.to_csv('test_lon_lat_predicted.csv')

etime = float(time.time()-stime)
print('Time:', etime)