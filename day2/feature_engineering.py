#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:17:10 2020

@author: roshan
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('train.csv')

# Droping name, cabin and index column
dataset = dataset.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

#Ecoding gender to binary value
label_encoder = LabelEncoder()
dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])

#One hot encoding for Pclass
'''one_hot = OneHotEncoder(categorical_features = [1])
dataset = one_hot.fit_transform(dataset).toarray()
'''
dataset = pd.get_dummies(dataset, prefix=['Class'], columns=['Pclass'], drop_first=False)
#nan in 'Age'
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean')
dataset['Age'] = imputer.fit_transform(dataset.iloc[:, 2:3])

# Embarked has 2 NaNs
replace = dataset['Embarked'].value_counts().argmax()
dataset['Embarked'].fillna(replace, inplace=True)

# dropping ticket number as no useful information
dataset = dataset.drop(['Ticket'], axis=1)

#Oneshot Embarked
dataset = pd.get_dummies(dataset, prefix=['Embarked'], columns=['Embarked'], drop_first=False)
final = dataset

# Few fares 0


