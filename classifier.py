import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#load the diabetics data set
X,Y = datasets.load_diabetes(return_X_y=True)

#split the dataset into test and training data
train_x, test_x, train_y, test_y = train_test_split(X,Y)

#initialize a Random forest classifier with 
# criteria as entropy, 
# max depth of decision trees as 10
# max features in each decision tree be selected automatically
rf = RandomForestClassifier(n_estimators=10, 
        criterion='entropy', 
        max_depth=10, 
        max_features='auto', 
        bootstrap=True,
        oob_score=True,
        verbose=1)

#fit the data        
rf.fit(train_x, train_y)

#print the feature importance - tbd
print (rf.feature_importances_)

#print the oob-score (out of box features error score)
print (rf.oob_score_)

#do a prediction on the test X data set
predicted_y = rf.predict(test_x)

#print the confusion matrix
print (confusion_matrix(test_y, predicted_y))

#print the classification report
print (classification_report(test_y, predicted_y))