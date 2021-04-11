import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB,MultinomialNB

data=pd.read_csv("train.csv")
X=data.iloc[:,range(2,28)].values
y=data.iloc[:,1].values

l=[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]
for i in l:
	data['var'+str(i-1)]=data['var'+str(i-1)].astype(str)

X_dum=data.iloc[:,[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]]
X_num=data.iloc[:,[6,10,16,18,19,20,21,23]]
X_dum=pd.get_dummies(X_dum)
del X_dum['var1_-1']
classifier1=GaussianNB()
classifier2=MultinomialNB()
model1=classifier1.fit(X_num,y)
model2=classifier2.fit(X_dum,y)

test=pd.read_csv("test.csv")
for i in l:
	test['var'+str(i-1)]=test['var'+str(i-1)].astype(str)

X_test_dum=test.iloc[:,[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]]
X_test_num=test.iloc[:,[6,10,16,18,19,20,21,23]]
X_test_dum=pd.get_dummies(X_test_dum)

for i in X_test_dum.columns:
	if i not in X_dum:
		del X_test_dum[i]

pred1=model1.predict_proba(X_test_num)[:,1]
pred2=model2.predict_proba(X_test_dum)[:,1]
# Import predictions of XGBoost
import pickle
pred3=pickle.load(open('xgb.pkl','rb'))

# Normalize probabilities
max1,max2,max3,min1,min2,min3=max(pred1),max(pred2),max(pred3),min(pred1),min(pred2),min(pred3)
pred1=[(i-min1)/(max1-min1) for i in pred1]
pred2=[(i-min2)/(max2-min2) for i in pred2]
pred3=[(i-min3)/(max3-min3) for i in pred3]

pred1=np.array(pred1).reshape(len(pred1),1)
pred2=np.array(pred2).reshape(len(pred2),1)
pred3=np.array(pred3).reshape(len(pred3),1)
prediction=np.hstack((pred1,pred2,pred3))

# ENSEMBLE
# model imported
model=pickle.load(open('naive_model.pkl','rb'))
pred=model.predict_proba(prediction)[:,1]

submission=pd.DataFrame()
submission['Resident ID']=test['Resident ID']
submission['Disease Flag']=pred
# Final submission file
submission.to_csv('submission_final.csv',index=False)