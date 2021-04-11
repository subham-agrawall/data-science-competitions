# Rare Disease Prediction
## ZS case study for Young Data Scientist 2017
This competition was conducted by ZS Associates. Detailed problem statement is present in Problem Statement.pptx

# 1st rank model
<img src="https://github.com/subham-agrawall/data-science-competitions/blob/main/Rare%20Disease%20Prediction/solution.png" height="300" width="500">

# Code Flow
1. xgboost.py -   
    Trained XGBoost model and created submissions.   
    XGBoost model stored as a pickle file.
3. validation.py -  
    Train data is split into train and validation datasets.   
    Cross validation ROC-AUC scores are calculated for all models i.e. XGBoost, Multinomial Naive Bayes and Gaussian Naive Bayes.   
    Stacked gaussian model is also validated and stored as a pickle file.  
3. submission.py -   
    Generates rank 1 submission file as per the above architecture.  
    XGBoost and stacked gaussian model are loaded as pickles.   
    Naive Bayes models are trained in this file.   
    
 # Experimentations
 1. trials/mlp.py -   
      Multi Layer Perceptron implemented in keras.  
      Baseline, smaller and larger networks are implemented to find the best model.
 2. trials/undersampling.py -    
      Undersampling technique implemented with combination of all machine learning models.  
      Evaluated performance of all undersampling models.
 3. trails/balanced.py -  
      Oversampling technique SMOTE is implemented with SVM algorithm. 
