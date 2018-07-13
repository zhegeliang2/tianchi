import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
from sklearn.cross_validation import PredefinedSplit
import numpy as np
import pickle


dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

#print(predictors)
predictors = [x for x in dataset1.columns.tolist()if x.startswith("c_") or x.startswith("u_") or x.startswith("m_") or x.startswith("um_") or x.startswith("t_")]
print(predictors)
dataset12 = pd.concat([dataset1,dataset2],axis=0)
dataset1_y = dataset1.label
#dataset1_x = dataset1.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
dataset1_x = dataset1[predictors]


dataset2_y = dataset2.label
#dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset2_x = dataset2[predictors]

dataset12_y = dataset12.label
#dataset12_x = dataset12.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset12_x = dataset12[predictors]

dataset3_preds = dataset3[['User_id','Coupon_id','Date_received']]
#dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)
dataset3_x = dataset3[predictors]

print dataset1_x.shape,dataset2_x.shape,dataset3_x.shape

dataset1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
dataset2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
dataset12 = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

#train on dataset1, evaluate on dataset2
watchlist = [(dataset1,'train'),(dataset2,'val')]
model = xgb.train(params,dataset1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)

#watchlist = [(dataset12,'train')]
#model = xgb.train(params,dataset12,num_boost_round=3500,evals=watchlist)
val = pd.concat([dataset2_x, dataset2_y])
val["pred_prob"] = model.predict(val)
val.pred_prob= MinMaxScaler().fit_transform(val.pred_prob.reshape(-1, 1))
val_group = val.groupby(['Coupon_id'])
aucs = []
for i in val_group:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

#predict test set
dataset3_preds['label'] = model.predict(dataset3)
dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.reshape(-1, 1))
dataset3_preds.sort_values(by=['Coupon_id','label'],inplace=True)
dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
print(dataset3_preds.describe())
    
#save feature score
#feature_score = model.get_fscore()
#feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
#fs = []
#for (key,value) in feature_score:
#    fs.append("{0},{1}\n".format(key,value))
#    
#with open('xgb_feature_score.csv','w') as f:
#    f.writelines("feature,score\n")
#    f.writelines(fs)
#
