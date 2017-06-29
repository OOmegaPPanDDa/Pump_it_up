import pandas as pd
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score as cv_s
from bayes_opt import BayesianOptimization as BayesOpt

stime = time.time()
train=pd.read_csv('train_lon_lat_predicted.csv',index_col=0)
target=pd.read_csv('target.csv',index_col=0)

def xgbcv(max_depth, 
          max_features, 
          n_estimators,
          min_samples_leaf,
          n_jobs=-1):
    return cv_s(ExtraTreesClassifier(max_depth=int(max_depth),
                                 max_features=max_features,
                                 n_estimators=int(n_estimators),
                                 min_samples_leaf=int(min_samples_leaf)),
                    train,
                    target['status_group'],
                    "log_loss",
                    cv=4).mean() 
xgboostBO = BayesOpt(xgbcv,
                                 {
                                  'max_depth': (15,30),
                                  'max_features': (0.5, 1),
                                  'n_estimators': (100,401),
                                  'min_samples_leaf': (1,21),
                                  })                                

print ("Start Optimization of Main Model")
xgboostBO.maximize(init_points=20,n_iter=130, xi=0.1,  acq="poi")
print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
print("\n Parameters: %s" % bo.res['max']['max_params'])

          