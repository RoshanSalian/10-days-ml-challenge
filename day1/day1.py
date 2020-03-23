#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:47:51 2020

@author: roshan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../time_series_2019-ncov-Confirmed.csv')
print(dataset.shape) #(487, 65)

del_col = []
for column in dataset:
    print( ((dataset[column]==0).sum()/dataset.shape[0]) )
    if ((dataset[column]==0).sum()/dataset.shape[0]) >= 0.7:
        del_col.append(column)
    
print(del_col)

for _ in  del_col:
    del dataset[_]

'''Condition 1'''    
# new dataset.shape (487, 23)
country_wise = dataset.drop(['Lat', 'Long', 'Province/State'], axis=1)  
country_wise = country_wise.sort_values("Country/Region")
country_wise1 = country_wise.groupby("Country/Region").agg('sum')
sorted_country = country_wise1.sort_values('3/22/20', ascending=False)
final_country = sorted_country.iloc[:21, -1]
final_country = final_country.reset_index()

X = final_country['Country/Region']
y_pos = np.arange(len(X))
Y = final_country['3/22/20']
plt.bar(y_pos, Y, align='center', alpha=0.5)
plt.xticks(y_pos, X, rotation=45, fontsize=8)
plt.xlabel('Infected Cases')
plt.xlabel('Country')
plt.ylabel('Infected Cases')
plt.title('Top 20 Covid19 Infected Country(22nd March, 2020)')
plt.show()


'''Condition 2'''
italy = sorted_country.iloc[1, :]
italy = italy.reset_index()
X_i = italy['index'].str[:-3]
y_pos = np.arange(len(X_i))
Y_i = italy['Italy']
plt.bar(y_pos, Y_i, align='center', alpha=0.5)
plt.xticks(y_pos, X_i, fontsize=8)
plt.xlabel('Infected Cases')
plt.xlabel('Country')
plt.ylabel('Infected Cases')
plt.title('Infection count in Italy')
plt.show()

'''Condition 3'''
country = country_wise1.reset_index()
country = country.iloc[:, [0, 19]]

cc = pd.read_csv('Countries-Continents.csv')

merged = pd.merge(left=country, right=cc, left_on='Country/Region', right_on='Country')
merged_mod = merged.drop(['Country/Region', 'Country'], axis=1)
merged_mod = merged_mod.groupby("Continent").agg('sum')
merged_mod = merged_mod.reset_index()
X_m = merged_mod['Continent']
y_pos = np.arange(len(X_m))
Y_m = merged_mod['3/22/20']
plt.bar(y_pos, Y_m, align='center', alpha=0.5)
plt.xticks(y_pos, X_m, fontsize=8)
plt.xlabel('Infected Cases Continents')
plt.xlabel('Continent')
plt.ylabel('Infected Cases')
plt.title('Infection count in Continent')
plt.show()

