# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:52:52 2017

@author: HSIN
"""

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold

stime = time.time()

nclass = 3

train = np.loadtxt("./stack1_train.txt")
test = np.loadtxt("./stack1_test.txt")

target = pd.read_csv('./target.csv', index_col = 0)

nfold = 10

outcome = target['status_group']

## Classifiers


# XGB              
xgb_classifier = XGBClassifier(
                    max_depth = 7,
                    learning_rate = 0.02,
                    n_estimators = 200,
                    gamma = 0.08,
                    min_child_weight = 3,
                    subsample = 0.5,
                    colsample_bytree = 0.9,
                    reg_alpha = 0.2,
                    objective ='multi:softmax'
                    )
                  
              
# logistic regr
log_regr_classifier = LogisticRegression(
                                            C = 10 ** (3),
                                            max_iter = 800
                                            )

classifiers = [xgb_classifier, log_regr_classifier]



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
        xtrain, xtest = train[train_idx,:], train[test_idx,:]
        
        # validation/test data
        ytrain, ytest = outcome[train_idx], outcome[test_idx]
        
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
        



np.savetxt("stack2_train.txt", blend_train) 
np.savetxt("stack2_test.txt", blend_sub) 

print('Log Loss')
print(np.mean(log_loss_eval_rec, axis=0))
print('Acc')
print(np.mean(acc_eval_rec, axis=0))

etime = float(time.time()-stime)
print('Time:', etime)