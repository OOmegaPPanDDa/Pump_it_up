import pandas as pd
import time
import xgboost as xgb
from sklearn.cross_validation import cross_val_score as cv_s
from bayes_opt import BayesianOptimization as BayesOpt

train=pd.read_csv('train_lon_lat_predicted.csv.csv',index_col=0)
target=pd.read_csv('target.csv',index_col=0)

def xgbcv(max_depth,  
          learning_rate, 
          n_estimators,
          min_child_weight,
          gamma,
          subsample,
          colsample_bytree,
          reg_alpha, 
          reg_lambda,
          silent=True):
    return cv_s(xgb.XGBClassifier(max_depth=int(max_depth),
                                 learning_rate=learning_rate,
                                 n_estimators=int(n_estimators),
                                 gamma=gamma,
                                 reg_alpha=reg_alpha,
                                 min_child_weight=min_child_weight,
                                 objective='multi:softmax'),
                    train,
                    target['status_group'],
                    "accuracy",
                    cv=4).mean() 
## can optimize std
xgboostBO = BayesOpt(xgbcv,
                                 {
                                  'max_depth': (4,22),
                                  'learning_rate': (0.01, 0.2),
                                  'n_estimators': (200,600),
                                  'gamma': (0.01, 10),
                                  'min_child_weight': (1,40),
                                  'subsample': (0.2, 1),
                                  'colsample_bytree' :(0.2, 1),
                                  'reg_alpha':(0, 10),
                                  'reg_lambda':(0, 10)
                                  })                                

print ("Start Optimization of Main Model")
xgboostBO.maximize(init_points=10,n_iter=110, xi=0.0,  acq="poi")


