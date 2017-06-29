import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import pickle

stime = time.time()

nclass=3

trainb=np.loadtxt("train_stage1.csv")
testb=np.loadtxt("pred_stage1.csv")
target=pd.read_csv('target.csv',index_col=0)
nf=10

outcome=target['status_group']

cclf1=XGBClassifier(max_depth=7,
              	learning_rate= 0.02358,
              	n_estimators=189,
              	gamma=0.07479,
              	min_child_weight=3.0666,
              	subsample=0.49698,
              	colsample_bytree=0.9517,
              	reg_alpha=0.2065,
              	objective='multi:softmax')

              
cclf2=LogisticRegression(C=1200,max_iter=800)

cclfs=[cclf1, cclf2]

print ("Start Stacking Models")
eval_rec=np.zeros((nf,len(cclfs)))
blend_temp=np.zeros((trainb.shape[0],nclass))
blend_sub_temp=np.zeros((testb.shape[0],nclass))
blend_train=np.zeros((trainb.shape[0],nclass*len(cclfs)))
blend_sub=np.zeros((testb.shape[0],nclass*len(cclfs)))

for j, clf in enumerate(cclfs):
    print (str(j)+"th Classifier")
    ### K-Fold with Shufffle
    skf = list(StratifiedKFold(outcome, nf))
    for i in range(nf):
        train, test=skf[i]
        xtrain, xtest = trainb[train,:], trainb[test,:]
        ytrain, ytest = outcome[train], outcome[test]
        clf.fit(xtrain, ytrain)
        path = 'save/stage2clf'+str(j)+str(i)+'.pickle'
        file = open(path,'wb')
        pickle.dump(clf,file)
        ytest_pred = clf.predict_proba(xtest)
        blend_temp[test]=ytest_pred
        sub_pred = clf.predict_proba(testb)
            
        if i==0:
            blend_sub_temp=sub_pred
        else:
            blend_sub_temp=blend_sub_temp+sub_pred
                
        print (i, log_loss(ytest,ytest_pred), (time.time()-stime))
        eval_rec[i,j]=log_loss(ytest,ytest_pred)
    blend_train[:,nclass*j:nclass*j+nclass]=blend_temp
    blend_sub[:,nclass*j:nclass*j+nclass]=blend_sub_temp/float(nf)
        
np.savetxt("train_stage2.csv",blend_train) 
np.savetxt("pred_stage2.csv",blend_sub) 
np.savetxt("outcome.txt",outcome) 
