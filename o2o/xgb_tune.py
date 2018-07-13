import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve,auc
from sklearn.cross_validation import PredefinedSplit


def modelfit(alg, dtrain, predictors,target="label",useTrainCV=True, cv_folds=3, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', verbose_eval=True)
        print("estimator num:", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

#prepare
#train = pd.read_csv("../../../../data.csv")
dataset1 = pd.read_csv("./data/dataset1.csv")
#dataset1 = dataset1[dataset1.label!=-1]
dataset1.label.replace(-1,0, inplace=True)
dataset1= dataset1.fillna(0)
dataset2 = pd.read_csv("./data/dataset2.csv")
#dataset2 = dataset2[dataset2.label!=-1]
dataset2.label.replace(-1,0, inplace=True)
dataset2= dataset2.fillna(0)
dataset3 = pd.read_csv("./data/dataset3.csv")
dataset3= dataset3.fillna(0)
dataset12 = pd.concat([dataset1,dataset2],axis=0)

#print(predictors)
predictors = [x for x in dataset1.columns.tolist()if x.startswith("c_") or x.startswith("u_") or x.startswith("m_") or x.startswith("um_") or x.startswith("t_")]
print(predictors)

def check_xgb_model(train, valid, predictors):
    
    classifier = lambda:XGBClassifier(
     objective='binary:logistic',
     silent=True,
     booster='gbtree',
     learning_rate =0.1,
     n_estimators=300,
     max_depth=5,
     min_child_weight=2,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     scale_pos_weight=1,
     n_jobs=20,
     reg_alpha = 0,
     reg_lambda = 1,
     seed=100)

    model = Pipeline(steps=[
        ('en', classifier())
    ])

    parameters = {
        #'en__n_estimators':[100, 300, 500, 700, 1000],
        #'en__max_depth':range(3,10,2),
        #'en__min_child_weight':np.arange(1,2.5,0.1),
        #'en__gamma':[i/10.0 for i in range(0,6)]
        #'en__subsample':[i/100.0 for i in range(75,90, 5)],
        #'en__colsample_bytree':[i/100.0 for i in range(75, 90, 5)]
        #'en__reg_alpha':[1e-5, 1e-2, 0.1, 0, 1, 10, 100],
        #'en__reg_lambda':[1e-5, 1e-2, 0.1, 0, 1, 10, 100],
    }
    data = pd.concat([train, valid])
    print(data[predictors].head())
    print("train size:%s, val size:%s, data size:%s" %(train.shape[0], valid.shape[0], data.shape[0]))
    index = np.zeros(data.shape[0])
    index[:train.shape[0]] = -1
    ps = PredefinedSplit(test_fold=index)

    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=ps, 
        n_jobs=-1, 
        verbose=1,
        scoring='roc_auc')
    grid_search = grid_search.fit(data[predictors], 
                                  data['label'])
   
    return grid_search

model = check_xgb_model(dataset1, dataset2, predictors)
print(model.best_score_)
print(model.best_params_)

#model= XGBClassifier(
# objective='binary:logistic',
# silent=True,
# booster='gbtree',
# learning_rate =0.01,
# n_estimators=300,
# max_depth=5,
# min_child_weight=2,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# scale_pos_weight=1,
# n_jobs=20,
# reg_alpha = 0,
# reg_lambda = 1,
# seed=100)
#
#model.fit(dataset12[predictors], dataset12["label"], eval_metric='auc')

dataset2['pred_prob'] = model.predict_proba(dataset2[predictors])[:,1]
dataset2 = dataset2.groupby(['Coupon_id'])
aucs = []
for i in dataset2:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

y_test_pred= model.predict_proba(dataset3[predictors])[:,1]
t1 = dataset3[['User_id','Coupon_id','Date_received']].copy()
t1['Probability'] = y_test_pred.reshape(-1, 1)
t1.to_csv('sample_submission1.csv', index=False, header=False)
