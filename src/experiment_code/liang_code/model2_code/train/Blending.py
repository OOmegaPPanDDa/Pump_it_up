import pandas as pd
from xgboost.sklearn import XGBClassifier
import operator
import numpy as np
import pickle

train=np.loadtxt("train_stage2.csv")
test=np.loadtxt("pred_stage2.csv")
target=pd.read_csv('target.csv',index_col=0)
submission=pd.read_csv('SubmissionFormat.csv')

est=XGBClassifier(max_depth=7,
              	learning_rate= 0.02358,
              	n_estimators=189,
              	gamma=0.07479,
              	min_child_weight=3.0666,
              	subsample=0.4970,
              	colsample_bytree=0.9517,
              	reg_alpha=0.2065,
              	objective='multi:softmax')



est.fit(train,target['status_group'])
path = 'save/est.pickle'
file = open(path,'wb')
pickle.dump(est,file)
pred=est.predict(test)
importances = est.booster().get_fscore()
sorted_imp = sorted(importances.items(), key=operator.itemgetter(1))

output=np.chararray(len(pred), itemsize=30)
output[pred==0]='functional'
output[pred==1]='functional needs repair'
output[pred==2]='non functional'


submission['status_group']=output
submission.to_csv('output.csv',index=False)
            
