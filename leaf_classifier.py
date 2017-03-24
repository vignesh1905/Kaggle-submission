# -*- coding: utf-8 -*-
"""
Leaf Classification
"""

'''Leaf Classification for Kaggle submission
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

#Calculate the start time
startTime = datetime.now()

#np.random.seed(42)

#Importing the CSV to a dataframe
data_in  = pd.read_csv('/Users/vigneshk/Desktop/Kaggle/LeafClassification/train.csv',
                       header = 0)

data_test_in = pd.read_csv('/Users/vigneshk/Desktop/Kaggle/LeafClassification/test.csv',
                       header = 0)

#Dropping the non-integer columns
data_used  = data_in.drop(['id','species'],axis = 1).values
data_used_test  = data_test_in.drop(['id'],axis = 1).values

#Creating Labels for classification
from sklearn import preprocessing
encode = preprocessing.LabelEncoder()
encode.fit(data_in.species)
labels = encode.fit_transform(data_in.species)

#Trying Polynomial features(Did not improve accuracy overfitted the data)
#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(degree = 2)
#poly.fit(data_used)
#data_train_poly  = poly.fit_transform(data_used)
#data_test_poly = poly.fit_transform(data_used_test)

#Preprocessing the data
scaler = StandardScaler()
scaler.fit(data_used)
data_train = scaler.fit_transform(data_used)

#scaler.fit(data_used_test)
data_test = scaler.fit_transform(data_used_test)

#Importing the required models
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver = 'lbfgs',
                             multi_class = 'multinomial',
                             verbose = 1000000000)

#from sklearn.ensemble import RandomForestClassifier
#rand_forest = RandomForestClassifier()

#Using GridSearchCV for findig the parameters
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[1400,1450,1500,1600,1700,1800,1900], 'tol': [0.00001, 0.001, 0.0001, 0.005]}
clf = GridSearchCV(log_reg, param_grid, scoring = 'log_loss', cv=5)
clf.fit(data_train, labels)

#Using the best params from grid search to use it in the model
y_proba = clf.predict_proba(data_test)

#forest_proba = rand_forest.predict_proba(data_test)
#Saving the csv to be submitted
test_ids = data_test_in.id
submission = pd.DataFrame(y_proba, index=test_ids, columns=encode.classes_)
#submission.to_csv('/Users/vigneshk/Desktop/Kaggle/LeafClassification/log_reg_2.csv')

print datetime.now() - startTime
