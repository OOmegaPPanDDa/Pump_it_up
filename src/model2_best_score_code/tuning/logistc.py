import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score as cv_s
from bayes_opt import BayesianOptimization as BayesOpt

target=pd.read_csv('target.csv',index_col=0)
train=pd.read_hdf('train_one_hot.h5')

LogisticRegression(C=1.0,max_iter=800)

def lrcv(C):
    return cv_s(LogisticRegression(C=10**C),
                    train,
                    target['status_group'],
                    "log_loss",
                    cv=4).mean() 

xgboostBO = BayesOpt(lrcv,
                                 {
                                  'C': (-5,2)
                                  })                                

print ("Start Optimization of Main Model")
xgboostBO.maximize(init_points=20,n_iter=130, xi=0.1,  acq="poi")