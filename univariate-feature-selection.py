import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LogisticRegression

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

def chi2BasedFeatureSelection(X,Y,func):

    #Use the SelectKBest function to select the best 5 features.
    kBestFeatures = SelectKBest(score_func=func, k=5)

    #Fit the data
    fittedData=kBestFeatures.fit(X,Y)

    #Find the score of each feature and print it
    scores=fittedData.scores_
    print('Scores of each feature are, select the top N features')
    print(scores)

    #transform the input data X and generate transformed X for prediction.
    newX = fittedData.transform(X)
    print (newX[0:5,0:])

    model = LogisticRegression(solver='lbfgs', max_iter=1000)

    trainX, testX, trainY, testY = train_test_split(newX,Y)
    model.fit(trainX,trainY)
    predicted_Y=model.predict(testX)
    printModelPerformanceSummary(testY,predicted_Y)

names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("diabetes-data.csv",names=names)

X= data.iloc[:,0:8]
Y=data.iloc[:,8]

model = LogisticRegression(solver='lbfgs', max_iter=1000)

print ("**** Model performance before feature selection ******")
trainX, testX, trainY, testY = train_test_split(X,Y)
model.fit(trainX,trainY)
predicted_Y=model.predict(testX)
printModelPerformanceSummary(testY,predicted_Y)

print("******* Feature Selection Using SelectKBest class and chi2 function ********")
chi2BasedFeatureSelection(X,Y,chi2)

print("******* Feature Selection Using SelectKBest class and f_classif function ********")
chi2BasedFeatureSelection(X,Y,f_classif)