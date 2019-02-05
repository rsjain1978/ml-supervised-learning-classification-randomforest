import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def printModelPerformanceSummary(test_y, predicted_y):
    #print the confusion matrix
    #confusion_matrix = confusion_matrix(test_y, predicted_y)
    #print (confusion_matrix)

    print ('Accuracy score is',accuracy_score(test_y, predicted_y))

    print ('Recall score is', recall_score(test_y, predicted_y, average='weighted'))

    print ('Precision store is', precision_score(test_y, predicted_y, average='weighted'))

    print ("F1 score is", f1_score(test_y, predicted_y, average='weighted'))

    #print the classification report
    print (classification_report(test_y, predicted_y))

names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data=pd.read_csv('diabetes-data.csv',names=names)

X=data.iloc[:,0:8]
Y=data.iloc[:,8]

print (X.shape)

model = LogisticRegression(solver='lbfgs',max_iter=1000)
selector = RFE(model, n_features_to_select=6)
selector = selector.fit(X,Y)

print(selector.support_)
print(selector.ranking_)

newX = selector.transform(X)

trainX, testX, trainY, testY = train_test_split(X,Y)
model.fit(trainX,trainY)
predicted_Y=model.predict(testX)
printModelPerformanceSummary(testY,predicted_Y)

trainX, testX, trainY, testY = train_test_split(newX,Y)
model.fit(trainX,trainY)
predicted_Y=model.predict(testX)
printModelPerformanceSummary(testY,predicted_Y)