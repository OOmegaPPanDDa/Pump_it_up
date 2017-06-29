# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:25:36 2017

@author: HSIN
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder

stime = time.time()

"""
Read Data
"""
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
train_label = pd.read_csv('./data/train_label.csv')

# rbind train test data
alldata = pd.concat([train, test], axis=0, ignore_index = True)




# drop unneccesary features
alldata = alldata.drop([
                        # useless                        
                        'num_private', 
                        'scheme_name',
                        'date_recorded', 'recorded_by', 
                        
                        
                        # too detailed
                        'id', 
                        'wpt_name', 'subvillage',
                        
                        # to ber considered
#                        'installer', 'funder', 'ward',
                        
                        # replicated
                        'payment_type', 'management_group',
                        'quantity_group', 'quality_group',
                        'source_type', 'source_class', 'waterpoint_type_group',
                        'extraction_type_group','extraction_type_class'
                        
                      ], axis=1)
                      
                      
            
            

# replace wrong lon lat data by None
alldata.ix[alldata['longitude'] == 0, 'longitude'] = None
alldata.ix[alldata['latitude'] > -0.1, 'latitude'] = None

lon_median = np.median(alldata.ix[pd.notnull(alldata['longitude']),'longitude'])
lat_median = np.median(alldata.ix[pd.notnull(alldata['latitude']),'latitude'])


# replace missing value by Other
alldata.ix[alldata['funder'] == '0', 'funder'] = "Other"
alldata.ix[alldata['installer'] == '0', 'installer'] = "Other"
alldata.ix[alldata['installer'] == '-', 'installer'] = "Other"


# replace missing year by median year
# 2000 is the median of all data excluding 0s
alldata.ix[alldata['construction_year'] == 0, 'construction_year'] = 2000




# set right type for numeric data
alldata['population'] = alldata['population'].astype('float64')
alldata['gps_height'] = alldata['gps_height'].astype('float64') 
alldata['construction_year'] = alldata['construction_year'].astype('float64')





# Label Encoder
le = LabelEncoder()
null_count = np.zeros(alldata.shape[1])

# Label Encoding for feature
for i in range(alldata.shape[1]):
    
    # count missing value for each feature
    null_count[i] = sum(pd.isnull(alldata.ix[:,i]))
    
    # replace missing value by Other
    if (null_count[i] > 0) & (alldata.columns[i] != 'longitude') & (alldata.columns[i] != 'latitude'):
        alldata.ix[pd.isnull(alldata.ix[:,i]), i] = "Other"
        
    # Label Encoding
    if alldata.ix[:,i].dtypes == 'object':
        alldata.ix[:,i][alldata.ix[:,i] == True] = 'True'
        alldata.ix[:,i][alldata.ix[:,i] == False] = 'False'
        le.fit(alldata.ix[:,i])
        alldata.ix[:,i]=le.transform(alldata.ix[:,i])


# Label Encoding for output label
le.fit(train_label['status_group'])
train_label['status_group'] = le.transform(train_label['status_group'])



# all data back to train test
train = alldata.ix[0:train.shape[0]-1,:]
test = alldata.ix[train.shape[0]:alldata.shape[0],:]
        

# filter out area feature
area_feature = [
            'latitude', 'longitude', 'region', 'region_code',
            'district_code', 'lga', 'gps_height',
            'ward'
            ]
         

# create lon predict data
lon_train = alldata.ix[pd.notnull(alldata['longitude']), area_feature]
# get lon train
lon_target = lon_train['longitude']
lon_train = lon_train.drop(['longitude', 'latitude'], axis=1)
# get lon test
lon_test = alldata.ix[pd.isnull(alldata['longitude']), area_feature]
lon_test = lon_test.drop(['longitude', 'latitude'], axis=1)




# create lat predict data
lat_train = alldata.ix[pd.notnull(alldata['latitude']), area_feature]
# get lat train
lat_target = lat_train['latitude']
lat_train=lat_train.drop(['longitude', 'latitude'], axis=1)
# get lat test
lat_test=alldata.ix[pd.isnull(alldata['latitude']), area_feature]
lat_test=lat_test.drop(['longitude', 'latitude'], axis=1)







train.to_csv('train_lon_lat_missing.csv')
test.to_csv('test_lon_lat_missing.csv')
train_label.to_csv('target.csv')

# replace lon lat missing value by median
train.ix[pd.isnull(train['longitude']),'longitude'] = lon_median
test.ix[pd.isnull(test['longitude']),'longitude'] = lon_median
train.ix[pd.isnull(train['latitude']),'latitude'] = lat_median
test.ix[pd.isnull(test['latitude']),'latitude'] = lat_median
train.to_csv('train_lon_lat_median.csv')
test.to_csv('test_lon_lat_median.csv')


# drop lon lat
train = train.drop(['longitude', 'latitude'], axis=1)
test = test.drop(['longitude', 'latitude'], axis=1)
train.to_csv('train_lon_lat_dropped.csv')
test.to_csv('test_lon_lat_dropped.csv')                      


lat_train.to_csv('lat_train.csv')
lat_test.to_csv('lat_test.csv')
lat_target.to_csv('lat_target.csv')

lon_train.to_csv('lon_train.csv')
lon_test.to_csv('lon_test.csv')
lon_target.to_csv('lon_target.csv')


etime = float(time.time()-stime)
print('Time:', etime)