import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import random

test = pd.read_csv("test_lon_lat_predicted.csv")
train = pd.read_csv("train_lon_lat_predicted.csv")
train_labs = pd.read_csv("./data/train_label.csv")
submission = pd.read_csv("./data/SubmissionFormat.csv")

train = train.fillna(" ")
test = test.fillna(" ")

categories=[]
for x in train.columns.values:
    if train[x].dtype.name == "object":
        categories.append(x)
for cat in categories:
    dummies = pd.get_dummies(train[cat]).rename(columns=lambda x: cat + str(x))
    train = pd.concat([train, dummies], axis=1)
    train = train.drop([cat], axis=1)
    
    dummies = pd.get_dummies(test[cat]).rename(columns=lambda x: cat + str(x))
    test = pd.concat([test, dummies], axis=1)
    test = test.drop([cat], axis=1)

random.seed(1230)
clf = RandomForestClassifier(n_estimators=200,max_depth=25)
y_true, k = pd.factorize(train_labs["status_group"])
clf.fit(train, y_true)

y_preds = clf.predict(train)

cm = confusion_matrix(y_true, y_preds)
accuracy = round(np.trace(cm)/float(np.sum(cm)),6)
print("Train error {}".format(1-accuracy))
print("Train accuracy {}".format(accuracy))

preds = clf.predict(test)
pred=list(k[i] for i in preds)
pred_t=pd.DataFrame({'id': submission["id"], "status_group": pred})

pred_t.to_csv("test_label.csv",index=False)