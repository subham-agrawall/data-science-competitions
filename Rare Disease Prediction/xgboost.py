# Generate predictions from XGBoost classifier
# import libraries
import pandas as pd 
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier

data=pd.read_csv("train.csv")
y=data.iloc[:,1].values
X_num=data.iloc[:,range(2,28)]
l=[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]
for i in l:
	data['var'+str(i-1)]=data['var'+str(i-1)].astype(str)

X_dum=data.iloc[:,[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]]
X_dum=pd.get_dummies(X_dum)
X=pd.concat([X_num,X_dum],axis=1)
del X['var1_-1']

classifier=XGBClassifier(learning_rate =0.05,
		n_estimators=100,
		max_depth=1,
		min_child_weight=4,
		gamma=0.5,
		subsample=1,
		colsample_bytree=0.6,
		objective= 'binary:logistic',
		nthread=8,
		scale_pos_weight=1.5,
		seed=27)
model=classifier.fit(X,y)

test=pd.read_csv("test.csv")
X_test_num=test.iloc[:,range(2,28)]
l=[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]
for i in l:
	test['var'+str(i-1)]=test['var'+str(i-1)].astype(str)

X_test_dum=test.iloc[:,[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]]
X_test_dum=pd.get_dummies(X_test_dum)
X_test=pd.concat([X_test_num,X_test_dum],axis=1)
for i in X_test.columns:
	if i not in X:
		del X_test[i]

pred=model.predict_proba(X_test)[:,1]
submission=pd.DataFrame()
submission['Resident ID']=test['Resident ID']
submission['Disease Flag']=pred
submission.to_csv('submission1.csv',index=False)

## generate pickle file for predictions from xgboost
import pickle
pickle.dump(pred,open('xgb.pkl','wb'))

### Feature importance
# a=['var1','var2','var3','var4','var5', 'var6', 'var7', 'var8','var9','var10','var11',
#     'var12', 'var13', 'var14', 'var15', 'var16', 'var17', 'var18',
#     'var19', 'var20', 'var21', 'var22', 'var23', 'var24', 'var25','var26']
# featimp = pd.Series(model.feature_importances_).sort_values(ascending=False)
# print(featimp) 