import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

#load the iris data set
X,Y = datasets.load_iris(return_X_y=True)

#split the dataset into test and training data
train_x, test_x, train_y, test_y = train_test_split(X,Y)

#Check the shape of test and training data
print ('Shape of training independent variable/predictor/features/attributes/variables ', train_x.shape)
print ('Shape of training dependent variable/class variable ', train_y.shape)
print ('Shape of testing X ', test_x.shape)
print ('Shape of testing Y ', test_y.shape)

#initialize a Random forest classifier with 
# 1000 decision trees or estimators
# criteria as entropy, 
# max depth of decision trees as 10
# max features in each decision tree be selected automatically
rf = RandomForestClassifier(n_estimators=1000,
        max_depth=10, 
        max_features='auto', 
        bootstrap=True,
        oob_score=True)

#fit the data        
rf.fit(train_x, train_y)

#print the feature importance - tbd
print ('Feature Importance is ',rf.feature_importances_)

#print the oob-score (out of box features error score)
print ('Out of box features score is ',rf.oob_score_)

#do a prediction on the test X data set
predicted_y = rf.predict(test_x)

errors = abs(predicted_y-test_y)
print ('Mean absolute error (MAE) ', round(np.mean(errors),2))

#print the confusion matrix
confusion_matrix = confusion_matrix(test_y, predicted_y)
print (confusion_matrix)

print ('Accuracy score is',accuracy_score(test_y, predicted_y))

print ('Recall score is', recall_score(test_y, predicted_y, average='weighted'))

print ('Precision store is', precision_score(test_y, predicted_y, average='weighted'))

print ("F1 score is", f1_score(test_y, predicted_y, average='weighted'))

#print the classification report
print (classification_report(test_y, predicted_y))