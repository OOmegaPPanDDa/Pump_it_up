import pandas as pd
#import time
import xgboost as xgb

train=pd.read_csv('./data/train_na.csv',index_col=0)
test=pd.read_csv('./data/test_na.csv',index_col=0)

lat_train_x=pd.read_csv('./data/lat_train_x.csv',index_col=0)
lat_test_x=pd.read_csv('./data/lat_test_x.csv',index_col=0)
lat_target=pd.read_csv('./data/lat_target.csv',index_col=0,header=None)

lon_train_x=pd.read_csv('./data/lon_train_x.csv',index_col=0)
lon_test_x=pd.read_csv('./data/lon_test_x.csv',index_col=0)
lon_target=pd.read_csv('./data/lon_target.csv',index_col=0,header=None)

# lon
lon_est=xgb.XGBRegressor(max_depth=15,
                     learning_rate=0.064324939284938294,
                     n_estimators=360,
                     gamma=0.03,
                     min_child_weight=1.42094958382,
                     subsample=0.9,
                     colsample_bytree=0.9)

lon_est.fit(lon_train_x,lon_target) 
lonpred=lon_est.predict(lon_test_x) 

# lat
lat_est=xgb.XGBRegressor(max_depth=15,
                     learning_rate=0.0684920938403,
                     n_estimators=285,
                     gamma=0.03,
                     min_child_weight=0.5893729938222,
                     subsample=0.9,
                     colsample_bytree=0.9)

lat_est.fit(lat_train_x,lat_target) 
latpred=lat_est.predict(lat_test_x)

# get train test index
train_ind = lon_test_x.index[lon_test_x.index < train.shape[0]]
test_ind = lon_test_x.index[lon_test_x.index >= train.shape[0]]


# replace train lon lat by predicted value
train.ix[train_ind,'longitude'] = lonpred[0:len(train_ind)] 
train.ix[train_ind,'latitude' ]= latpred[0:len(train_ind)]

# replace test lon lat by predicted value
test.ix[test_ind,'longitude'] = lonpred[len(train_ind):lonpred.shape[0]] 
test.ix[test_ind,'latitude'] = latpred[len(train_ind):latpred.shape[0]]

      

train.to_csv('./data/train_lon_lat_predicted.csv')
test.to_csv('./data/test_lon_lat_predicted.csv')