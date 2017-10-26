#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

#Add libraries
import numpy as np #Maths library
import matplotlib.pyplot as plt #Plot charts
import pandas as pd #Import and manage data sets

#Import the data set
dataset = pd.read_csv('Data.csv')
#Independent variables matrix
X = dataset.iloc[:, :-1].values #Left of [] is lines and right is columns
#Dependent variable vector
Y = dataset.iloc[:,3].values

#Missing data // Scikitlearn
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
#imputer = imputer.fit(X[:, 1:3]) #upper-bound is excluded
#X[:,1:3] = imputer.transform(X[:,1:3])

#Encode categorical variables
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Problem => not order between those encoded variables so we need to use dummy variables to convert
# this cat. variable in 3 (number of categories) new columns => 1 per variable category and feed with
# 0 or 1
#onehotencoder = OneHotEncoder(categorical_features =  [0])
#X = onehotencoder.fit_transform(X).toarray()

#Don't need to  use OneHotEncoder as this is the dependent variable
#labelencoder_y = LabelEncoder()
#Y = labelencoder_y.fit_transform(Y)

#Splitting data set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Feature scaling => needed to set same scale [-1; 1] as Eucledian distance is sum of squared coordinates 
# Or for performance (decision trees not based on Eucledian distances => much faster with feature scaling)
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test) #No need to fit for the test set
# not needed yor Y as classification with 2 categories
