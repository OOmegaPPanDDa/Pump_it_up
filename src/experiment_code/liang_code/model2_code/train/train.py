import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import pickle
from sklearn.linear_model import LogisticRegression

nclass=3

stime = time.time()

trainc=pd.read_csv('./data/train_lon_lat_predicted.csv',index_col=0)
testc=pd.read_csv('./data/test_lon_lat_predicted.csv',index_col=0)
target=pd.read_csv('./data/target.csv',index_col=0)
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
        path = './save/clf'+str(j)+str(i)+'.pickle'
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
        
np.savetxt("./data/train_stage1.csv",blend_train) 
np.savetxt("./data/pred_stage1.csv",blend_sub) 
np.savetxt("./data/outcome.txt",outcome) 

stime = time.time()

nclass=3

trainb=np.loadtxt("./data/train_stage1.csv")
testb=np.loadtxt("./data/pred_stage1.csv")
target=pd.read_csv('./data/target.csv',index_col=0)
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
        path = './save/stage2clf'+str(j)+str(i)+'.pickle'
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
        
np.savetxt("./data/train_stage2.csv",blend_train) 
np.savetxt("./data/pred_stage2.csv",blend_sub) 
np.savetxt("./data/outcome.txt",outcome) 


train=np.loadtxt("./data/train_stage2.csv")
test=np.loadtxt("./data/pred_stage2.csv")
target=pd.read_csv('./data/target.csv',index_col=0)
submission=pd.read_csv('./data/SubmissionFormat.csv')

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
path = './save/est.pickle'
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
submission.to_csv('./data/output.csv',index=False)
            

