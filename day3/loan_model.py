#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:14:51 2020

@author: roshan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('loan_status_train.csv')

'''
CHANGES MADE TO DATASET
1. Replaced Nan in Gender with most frequent
2. Married had NaNs
3. Dependents had 3+ and NaNs
4. Sel_emplyed had NaNs
5. Credit History
'''

# checking consistency
print(dataset['Gender'].unique())
dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode().iloc[0]) 

print(dataset['Married'].unique())
dataset['Married'] = dataset['Married'].fillna(dataset['Married'].mode().iloc[0])

dataset['Dependents'] = dataset['Dependents'].replace({'3+':'3'}) 
dataset['Dependents'] = dataset['Dependents'].fillna(dataset['Dependents'].mode().iloc[0])

dataset['Self_Employed'] = dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode().iloc[0])
dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].mode().iloc[0])

dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean())
# dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())

print(dataset.columns)
print(dataset['Loan_Status'].unique())

print(dataset.isnull().sum())

# Converting categorical value to numerical
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
onehot = LabelBinarizer()
dataset['Gender'] = onehot.fit_transform(dataset['Gender'])
dataset['Married'] = onehot.fit_transform(dataset['Married'])
dataset['Education'] = onehot.fit_transform(dataset['Education'])
dataset['Self_Employed'] = onehot.fit_transform(dataset['Self_Employed'])
dataset['Loan_Status'] = onehot.fit_transform(dataset['Loan_Status'])
dataset['Dependents'] = dataset['Dependents'].astype(int)

dataset = dataset.drop(['Loan_ID'], axis=1)

print(dataset['Property_Area'].unique())
label = LabelEncoder()
dataset['Property_Area'] = label.fit_transform(dataset['Property_Area'])
# one_hot = OneHotEncoder(categorical_features=['Property_Area'], drop='first')
dataset = pd.get_dummies(dataset, prefix=['Property_Area'], columns=['Property_Area'], drop_first=True)
dataset[['Property_Area_1', 'Property_Area_2']] = dataset[['Property_Area_1', 'Property_Area_2']].astype(int)

Y = []
X = []
for _ in dataset.columns:
    if(_ == 'Loan_Status'):
        Y.append(_)
    else:
        X.append(_)

Xi = dataset[X]
Yi = dataset[Y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xi, Yi, test_size=0.33, random_state=42)

from sklearn import svm
modelSVM = svm.SVC(kernel='linear')
modelSVM.fit(X_train, y_train)
y_pred = modelSVM.predict(X_test)
y_pred.columns = ['Loan_Status']
y_pred = pd.DataFrame(y_pred)

y_test = y_test.reset_index()
y_test = y_test.drop(['index'], axis=1)
from sklearn import metrics
print(metrics.accuracy_score( y_test, y_pred)) #0.6502463054187192


modelSVM2 = svm.SVC(kernel='rbf')
modelSVM2.fit(X_train, y_train)
print(modelSVM2.score(X_test, y_test))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
print("acc "+str(model1.score(X_test, y_test))) 
# 0.7980295566502463

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
print(model2.score(X_test, y_test))
# 0.6945812807881774

from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier()
model3.fit(X_train, y_train)
print(model3.score(X_test, y_test))
# 0.5714285714285714

from sklearn.naive_bayes import GaussianNB
model4  = GaussianNB().fit(X_train, y_train)
print(model4.score(X_test, y_test))
# 0.7931034482758621



