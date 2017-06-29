import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import pickle

nclass=3

stime = time.time()

trainc=pd.read_csv('train_lon_lat_predicted.csv',index_col=0)
testc=pd.read_csv('test_lon_lat_predicted.csv',index_col=0)
target=pd.read_csv('target.csv',index_col=0)
nf=10

outcome=target['status_group']       
cclf1=XGBClassifier(max_depth=14,
                    learning_rate=0.0588,
                    n_estimators=250,
                    objective='multi:softprob',
                    nthread=8,
                    gamma=0.6890,
                    min_child_weight=7.6550,
                    subsample=0.8, 
                    colsample_bytree=0.8)

              
cclf2=XGBClassifier(max_depth=15,
                    learning_rate=0.03599,
                    n_estimators=385,
                    objective='multi:softprob',
                    nthread=8,
                    gamma=0.6836,
                    min_child_weight= 4.3704,
                    subsample=0.8, 
                    colsample_bytree=0.8)

cclf3 = RandomForestClassifier(n_estimators=384, 
                               min_samples_leaf=2, 
                               max_features=0.5060, 
                               max_depth=26)


cclf4 = RandomForestClassifier(n_estimators=346, 
                               min_samples_leaf=5, 
                               max_features=0.5112, 
                               max_depth=25)                             

cclf5 = ExtraTreesClassifier(n_estimators=341,
                     		min_samples_split=5,
                     		max_features=0.7769,
                    		max_depth=25) 


cclf6 = ExtraTreesClassifier(n_estimators=387,
                     		min_samples_split=3,
                     		max_features=0.6636,
                    		max_depth=25)
                            

cclfs=[cclf1, cclf2, cclf3, cclf4, cclf5, cclf6]

print ("Start Stacking Models")
eval_rec=np.zeros((nf,len(cclfs)))
blend_temp=np.zeros((trainc.shape[0],nclass))
blend_sub_temp=np.zeros((testc.shape[0],nclass))
blend_train=np.zeros((trainc.shape[0],nclass*len(cclfs)))
blend_sub=np.zeros((testc.shape[0],nclass*len(cclfs)))

for j, clf in enumerate(cclfs):
    print (str(j)+"th Classifier")
    skf = list(StratifiedKFold(outcome, nf))
    for i in range(nf):
        train, test=skf[i]
        xtrain, xtest = trainc.ix[train,:], trainc.ix[test,:]
        ytrain, ytest = outcome.ix[train], outcome.ix[test]
        clf.fit(xtrain, ytrain)
        #clf.dump_model('clf'+str(j)+str(i)+'.txt')
        #clf.save_model('clf'+str(j)+str(i)+'.model')
        path = 'save/clf'+str(j)+str(i)+'.pickle'
        file = open(path,'wb')
        pickle.dump(clf,file)
        ytest_pred = clf.predict_proba(xtest)
        blend_temp[test]=ytest_pred
        sub_pred = clf.predict_proba(testc)
        if i==0:
            blend_sub_temp=sub_pred
        else:
            blend_sub_temp=blend_sub_temp+sub_pred
                
        print (i, log_loss(ytest,ytest_pred), (time.time()-stime))
        eval_rec[i,j]=log_loss(ytest,ytest_pred)
    blend_train[:,nclass*j:nclass*j+nclass]=blend_temp
    blend_sub[:,nclass*j:nclass*j+nclass]=blend_sub_temp/float(nf)
        
np.savetxt("train_stage1.csv",blend_train) 
np.savetxt("pred_stage1.csv",blend_sub) 
np.savetxt("outcome.txt",outcome) 
