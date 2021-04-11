import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
# %matplotlib inline
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # to split the data
from sklearn.cross_validation import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report,mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

norm=[6,10,16,18,19,20]
for i in norm:
	data['var'+str(i-1)]=StandardScaler().fit_transform(data['var'+str(i-1)].reshape(-1, 1))

X=data.iloc[:,range(2,28)].values
y=data.iloc[:,1].values

# Now lets check the class distributions
sns.countplot("Disease Flag",data=data)

# Explore data
for i in range(1,27):
	Disease = data[data["Disease Flag"]==1]
	Normal= data[data["Disease Flag"]==0]
	plt.figure(figsize=(10,6))
	plt.subplot(121)
	Disease['var'+str(i)].plot.hist()
	plt.subplot(122)
	Normal['var'+str(i)].plot.hist()
	plt.show()

# for undersampling we need a portion of majority class and will take whole data of minority class
# count fraud transaction is the total number of fraud transaction
# now lets us see the index of fraud cases
fraud_indices= np.array(data[data['Disease Flag']==1].index)
normal_indices = np.array(data[data['Disease Flag']==0].index)
# now let us a define a function for make undersample data with different proportion
# different proportion means with different proportion of normal classes of data
def undersample(normal_indices,fraud_indices,times):#times denote the normal data = times*fraud data
	Normal_indices_undersample = np.array(np.random.choice(normal_indices,(times*600),replace=False))
	undersample_data= np.concatenate([fraud_indices,Normal_indices_undersample])
	undersample_data = data.iloc[undersample_data,:]
	print("total number of record in resampled data is:",len(undersample_data))
	return(undersample_data)

## first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
	clf= model
	clf.fit(features_train,labels_train)
	pred=clf.predict(features_test)
	pred_prob=clf.predict_proba(features_test)[:,1]
	cnf_matrix=confusion_matrix(labels_test,pred)
	sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
	plt.show()
	fpr, tpr, thresholds = roc_curve(labels_test,pred_prob)
	plt.plot(fpr, tpr)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.rcParams['font.size'] = 12
	plt.title('ROC curve for syndrome classifier')
	plt.xlabel('False Positive Rate (1 - Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.grid(True)
	plt.show()
	print(roc_auc_score(labels_test,pred_prob))
	print(mean_squared_error(labels_test,pred_prob))
	colors = {0:'red', 1:'blue'}
	col=[colors[labels_test[i]] for i in range(len(labels_test))]
	plt.scatter(range(len(labels_test)),pred_prob,c=col)
	plt.show()

# calculate cross-validated AUC
def cross_validation(model,features,labels):
	results=cross_val_score(classifier, features, labels, cv=5, scoring='roc_auc')
	print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def data_preparation(data):
	X=data.iloc[:,range(2,28)].values
	y=data.iloc[:,1].values
	Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.3,random_state=42)
	return Xtr,Xts,ytr,yts

# Now make undersample data with differnt portion
# here i will take normal trasaction in  0..5 %, 0.66% and 0.75 % proportion of total data now do this for 
for i in range(1,4):
	print("the undersample data for {} proportion".format(i))
	Undersample_data = undersample(normal_indices,fraud_indices,i)
	print("------------------------------------------------------------")
	print("the model classification for {} proportion".format(i))
	undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test=data_preparation(Undersample_data)
	# clf= RandomForestClassifier(n_estimators=100)
	clf=LogisticRegression()
	model(clf,undersample_features_train,undersample_features_test,undersample_labels_train,undersample_labels_test)
	print("________________________________________________________________________________________________________")

featimp = pd.Series(clf.feature_importances_,index=data_features_train.columns).sort_values(ascending=False)
print(featimp) # this is the property of Random Forest classifier that it provide us the importance 
# of the features use
