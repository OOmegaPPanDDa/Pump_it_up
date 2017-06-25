# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 03:23:50 2017

@author: HSIN
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import time

stime = time.time()

train = pd.read_csv('./train_lon_lat_predicted.csv', index_col = 0)
test = pd.read_csv('./test_lon_lat_predicted.csv', index_col = 0)
target = pd.read_csv('./target.csv', index_col = 0)

object_col = [  'installer',
                'basin',
                'region',
                'lga',
                'ward',
                'public_meeting',
                'scheme_management',
                'permit',
                'extraction_type',
                'extraction_type_class',
                'management',
                'payment',
                'water_quality',
                'quantity',
                'source',
                'waterpoint_type'
                ]


train_obj = np.array(train.ix[:, [x in object_col for x in train.columns]])
train_numeric = np.array(train.ix[:, [x not in object_col for x in train.columns]])

test_obj = np.array(test.ix[:,[x in object_col for x in train.columns]])
test_numeric = np.array(test.ix[:,[x not in object_col for x in train.columns]])

all_obj = np.concatenate((train_obj, test_obj), axis = 0)

one_hot_encoder = OneHotEncoder()
all_obj = one_hot_encoder.fit_transform(all_obj).toarray()

train = np.concatenate((all_obj[0:train.shape[0],:], train_numeric), axis = 1)
test = np.concatenate((all_obj[train.shape[0]:all_obj.shape[0],:], test_numeric), axis = 1)

del(all_obj, test_obj, train_obj, train_numeric, test_numeric)

pd.DataFrame(train).to_hdf('train_one_hot.h5','w')
pd.DataFrame(test).to_hdf('test_one_hot.h5','w')

etime = float(time.time()-stime)
print('Time:', etime)