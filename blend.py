# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 00:31:20 2017

@author: HSIN
"""

import pandas as pd
import time
import xgboost as xgb
import numpy as np

stime = time.time()


train = np.loadtxt("stack2_train.txt")[:,0:6]
test = np.loadtxt("stack2_test.txt")[:,0:6]

target = pd.read_csv('./target.csv', index_col=0)

submission = pd.read_csv('./data/SubmissionFormat.csv')

xgb_classifier = xgb.XGBClassifier(
                        max_depth = 7,
                        learning_rate = 0.02,
                        n_estimators = 200,
                        gamma = 0.08,
                        min_child_weight = 3,
                        subsample = 0.5,
                        colsample_bytree = 0.9,
                        reg_alpha = 0.2,
                        objective = 'multi:softmax'
                        )
                 
xgb_classifier.fit(train, target['status_group'])
pred = xgb_classifier.predict(test)


# replace encoding label back
prediction = np.chararray(len(pred), itemsize = 30, unicode = True)
prediction[pred==0] = "functional"
prediction[pred==1] = "functional needs repair"
prediction[pred==2] = "non functional"

submission['status_group'] = prediction
submission.to_csv('prediction.csv', index = False)
            

etime = float(time.time()-stime)
print('Time:', etime)