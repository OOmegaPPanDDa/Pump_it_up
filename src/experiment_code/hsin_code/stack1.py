# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:29:44 2017

@author: HSIN
"""

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

stime = time.time()

nclass = 3


## situation1: drop lon lat
#train = pd.read_csv('./train_lon_lat_dropped.csv', index_col = 0)
#test = pd.read_csv('./test_lon_lat_dropped.csv', index_col = 0)


## situation2: median lon lat
#train = pd.read_csv('./train_lon_lat_median.csv', index_col = 0)
#test = pd.read_csv('./test_lon_lat_median.csv', index_col = 0)


# situation3: xgb predicted lon lat
#train = pd.read_csv('./train_lon_lat_xgb_predicted.csv', index_col = 0)
#test = pd.read_csv('./test_lon_lat_xgb_predicted.csv', index_col = 0)

# situation4: nn predicted lon lat
train = pd.read_csv('./train_lon_lat_nn_predicted.csv', index_col = 0)
test = pd.read_csv('./test_lon_lat_nn_predicted.csv', index_col = 0)


#drop_list = [
#                'funder',
#                'installer',
#                'scheme_management',
#                
#                'basin',
#                'region',
#                'region_code',
#                'district_code',
#                'lga',
#                'ward',
#                
#                'public_meeting',
#                'permit',
#                'payment'
#]
#
#train = train.drop(drop_list, axis=1)
#test = test.drop(drop_list, axis = 1)





### situation5: one hot
#train = pd.read_hdf('./train_one_hot.h5')
#test = pd.read_hdf('./test_one_hot.h5')



target = pd.read_csv('./target.csv', index_col = 0)


nfold = 10


outcome = target['status_group']

## Classifiers          
xgb_classifier = XGBClassifier(
                                max_depth = 15, 
                                learning_rate = 0.03,
                                n_estimators = 400, 
                                objective ='multi:softprob', 
                                nthread = 8,
                                gamma = 0.5,
                                min_child_weight = 5,
                                subsample = 0.8,
                                colsample_bytree = 0.8
                                )



rf_classifier = RandomForestClassifier(
                                        n_estimators = 500, 
                                        min_samples_leaf = 5, 
                                        max_features = 0.5, 
                                        max_depth = 25
                                        )   

ert_classifier = ExtraTreesClassifier(
                                        n_estimators = 400,
                                        min_samples_split = 3,
                                        max_features = 0.6,
                                        max_depth = 25
                                        )  
                            

classifiers = [xgb_classifier, rf_classifier, ert_classifier]
#classifiers = [rf_classifier, ert_classifier]


print ("Start Stacking Models")


log_loss_eval_rec = np.zeros((nfold, len(classifiers)))
acc_eval_rec = np.zeros((nfold, len(classifiers)))

# for each fold record
blend_temp = np.zeros((train.shape[0], nclass))
blend_sub_temp = np.zeros((test.shape[0], nclass))

# for all record
blend_train = np.zeros((train.shape[0], nclass * len(classifiers)))
blend_sub = np.zeros((test.shape[0], nclass * len(classifiers)))

for j, clf in enumerate(classifiers):
    
    print (str(j) + "th Classifier")
    
    # 10-fold
    skf = list(StratifiedKFold(outcome, nfold))
    
    for i in range(nfold):
        
        # train test split
        train_idx, test_idx = skf[i]
        
        # train data
        xtrain, xtest = train.ix[train_idx,:], train.ix[test_idx,:]
        
        # validation/test data
        ytrain, ytest = outcome.ix[train_idx], outcome.ix[test_idx]
        
        # fit classifier
        clf.fit(xtrain, ytrain)
        
        # validation
        ytest_pred = clf.predict_proba(xtest)
        blend_temp[test_idx] = ytest_pred
        
        # prediction
        sub_pred = clf.predict_proba(test)
            
        if i==0:
            blend_sub_temp = sub_pred
        else:
            blend_sub_temp = blend_sub_temp + sub_pred
        
        
        
        
        log_loss_value = log_loss(ytest, ytest_pred)
        acc_value = sum(np.argmax(ytest_pred, axis=1) == ytest)/len(ytest)
                
        print ('fold:', i)
        print ('log loss:', log_loss_value)
        print ('acc:', acc_value)
        print ('time:', time.time()-stime)
        
        
        # validation record
        log_loss_eval_rec[i, j] = log_loss_value
        acc_eval_rec[i, j] = acc_value
        
        
    # train prediction  
    blend_train[:, nclass * j:nclass * j + nclass] = blend_temp
    
    # test prediction
    blend_sub[:, nclass * j:nclass * j + nclass] = blend_sub_temp/float(nfold)
        



np.savetxt("stack1_train.txt", blend_train) 
np.savetxt("stack1_test.txt", blend_sub) 

print('Log Loss')
print(np.mean(log_loss_eval_rec, axis=0))
print('Acc')
print(np.mean(acc_eval_rec, axis=0))

etime = float(time.time()-stime)
print('Time:', etime)