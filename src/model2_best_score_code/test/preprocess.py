import pandas as pd
import numpy as np
#import time
from sklearn.preprocessing import LabelEncoder

train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
outcome=pd.read_csv('./data/train_label.csv')

trainum=train.shape[0]
alldata=pd.concat([train,test],axis=0,ignore_index=True)

alldata=alldata.drop(['id','num_private','waterpoint_type_group','source_type','payment_type','extraction_type_group',
            'quantity_group','management_group','quality_group','wpt_name','scheme_name','source_class',
            'date_recorded','recorded_by','subvillage'],axis=1)  
            
alldata.ix[alldata['latitude']>-0.1,'latitude']=None
alldata.ix[alldata['longitude']==0,'longitude']=None
alldata.ix[alldata['funder']=='0','funder']="Other"
alldata.ix[alldata['installer']=='0','installer']="Other"
alldata.ix[alldata['installer']=='-','installer']="Other"


alldata['population']=alldata['population'].astype('float64')           
#alldata['num_private']=alldata['num_private'].astype('float64') 
alldata['gps_height']=alldata['gps_height'].astype('float64') 
alldata['construction_year']=alldata['construction_year'].astype('float64')





le=LabelEncoder()
nullct=np.zeros(alldata.shape[1])
for i in range(alldata.shape[1]):
    nullct[i]=sum(pd.isnull(alldata.ix[:,i]))    
    if (nullct[i]>0) & (alldata.columns[i]!='longitude') & (alldata.columns[i]!='latitude'):
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]="Other"   
    if alldata.ix[:,i].dtypes=='object':
        alldata.ix[:,i][alldata.ix[:,i] == True] = 'True'
        alldata.ix[:,i][alldata.ix[:,i] == False] = 'False'
        le.fit(alldata.ix[:,i])
        alldata.ix[:,i]=le.transform(alldata.ix[:,i])   

le.fit(outcome['status_group'])
outcome['status_group']=le.transform(outcome['status_group'])

train=alldata.ix[0:trainum-1,:]
test=alldata.ix[trainum:alldata.shape[0],:]
        
areacol=['latitude','longitude','region','region_code','district_code','lga','ward','gps_height']
lontrain=alldata.ix[pd.notnull(alldata['longitude']),areacol]
lontarget=lontrain['longitude']
lontrain=lontrain.drop(['longitude','latitude'],axis=1)
lontest=alldata.ix[pd.isnull(alldata['longitude']),areacol]
lontest=lontest.drop(['longitude','latitude'],axis=1)

lattrain=alldata.ix[pd.notnull(alldata['latitude']),areacol]
lattarget=lattrain['latitude']
lattrain=lattrain.drop(['longitude','latitude'],axis=1)
lattest=alldata.ix[pd.isnull(alldata['latitude']),areacol]
lattest=lattest.drop(['longitude','latitude'],axis=1)

train.to_csv('./data/train_na.csv')
test.to_csv('./data/test_na.csv')
outcome.to_csv('./data/target.csv')

lattrain.to_csv('./data/lat_train_x.csv')
lattest.to_csv('./data/lat_test_x.csv')
lattarget.to_csv('./data/lat_target.csv')

lontrain.to_csv('./data/lon_train_x.csv')
lontest.to_csv('./data/lon_test_x.csv')
lontarget.to_csv('./data/lon_target.csv')
