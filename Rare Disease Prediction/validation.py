import pandas as pd 
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from xgboost.sklearn import XGBClassifier

data=pd.read_csv("train.csv")
X=data.iloc[:,range(2,28)].values
y=data.iloc[:,1].values

X_num1=data.iloc[:,range(2,28)]
l=[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]
for i in l:
	data['var'+str(i-1)]=data['var'+str(i-1)].astype(str)

X_dum=data.iloc[:,[2,3,4,5,7,8,9,11,12,13,14,15,17,22,24,25,26,27]]
X_num=data.iloc[:,[6,10,16,18,19,20,21,23]]
X_dum=pd.get_dummies(X_dum)
del X_dum['var1_-1']
X_xgb=pd.concat([X_num1,X_dum],axis=1)

Xtr1,Xts1,ytr1,yts1=train_test_split(X_num,y,test_size=0.3,random_state=42)
Xtr2,Xts2,ytr2,yts2=train_test_split(X_dum,y,test_size=0.3,random_state=42)
Xtr3,Xts3,ytr3,yts3=train_test_split(X_xgb,y,test_size=0.3,random_state=42)

classifier1=GaussianNB()
classifier2=MultinomialNB()
classifier3=XGBClassifier(learning_rate =0.05,
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
model1=classifier1.fit(Xtr1,ytr1)
model2=classifier2.fit(Xtr2,ytr2)
model3=classifier3.fit(Xtr3,ytr3)

Ypred_prob1=model1.predict_proba(Xts1)[:,1]
Ypred_prob2=model2.predict_proba(Xts2)[:,1]
Ypred_prob3=model3.predict_proba(Xts3)[:,1]

# Model1 evaluation ################################################
fpr1, tpr1, thresholds1 = metrics.roc_curve(yts1, Ypred_prob1)
plt.plot(fpr1, tpr1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for GaussianNB')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
# print(metrics.roc_auc_score(yts1, Ypred_prob1))
# colors = {0:'white', 1:'blue'}
# col=[colors[yts1[i]] for i in range(len(yts1))]
# plt.scatter(range(len(yts1)),Ypred_prob1,c=col)
# plt.show()
# calculate cross-validated AUC
results=cross_val_score(classifier1, X_num, y, cv=10, scoring='roc_auc')
print("Results for GaussianNB: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Model2 evaluation ################################################
fpr2, tpr2, thresholds2 = metrics.roc_curve(yts2, Ypred_prob2)
plt.plot(fpr2, tpr2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for MultinomialNB')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
# print(metrics.roc_auc_score(yts2, Ypred_prob2))
# colors = {0:'white', 1:'blue'}
# col=[colors[yts2[i]] for i in range(len(yts2))]
# plt.scatter(range(len(yts2)),Ypred_prob2,c=col)
# plt.show()
# calculate cross-validated AUC
results=cross_val_score(classifier2, X_dum, y, cv=10, scoring='roc_auc')
print("Results for MultinomialNB: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Model3 evaluation #####################################################
fpr3, tpr3, thresholds3 = metrics.roc_curve(yts3, Ypred_prob3)
plt.plot(fpr3, tpr3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for XBG')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
# print(metrics.roc_auc_score(yts3, Ypred_prob3))
# colors = {0:'white', 1:'blue'}
# col=[colors[yts3[i]] for i in range(len(yts3))]
# plt.scatter(range(len(yts3)),Ypred_prob3,c=col)
# plt.show()
# calculate cross-validated AUC
results=cross_val_score(classifier3, X_xgb, y, cv=10, scoring='roc_auc')
print("Results for XGB: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# ANALYZE ##############################################################################
# Normalize probabilites of all the 3 models
pred1,pred2,pred3=Ypred_prob1,Ypred_prob2,Ypred_prob3
max1,max2,max3,min1,min2,min3=max(pred1),max(pred2),max(pred3),min(pred1),min(pred2),min(pred3)
pred1=[(i-min1)/(max1-min1) for i in pred1]
pred2=[(i-min2)/(max2-min2) for i in pred2]
pred3=[(i-min3)/(max3-min3) for i in pred3]
### 3-D scatter plot of probabilities
# fig = plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(fig)
# ax.scatter(pred1,pred2,pred3,c=col)
# plt.show()

# ENSEMBLE-Introduce classifier4 #############################################################
pred1=np.array(pred1).reshape(len(pred1),1)
pred2=np.array(pred2).reshape(len(pred2),1)
pred3=np.array(pred3).reshape(len(pred3),1)
prediction=np.hstack((pred1,pred2, pred3))
yts=yts1
classifier4=GaussianNB()
results=cross_val_score(classifier4, prediction, yts, cv=10, scoring='roc_auc')
print("Results for Ensemble GaussianNB: %.2f%%" % (results.mean()*100))

#####pickle file
model4=classifier4.fit(prediction,yts)
import pickle
pickle.dump(model4,open('naive_model.pkl','wb'))