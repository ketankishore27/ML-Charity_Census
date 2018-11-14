# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:55:48 2018

@author: kekishor
"""

# ---------------------> IMporting the required libraries ------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------> Getting the required data ------------------------------------

data = pd.read_csv('census.csv')

# ---------------------> Length of the dataset ----------------------------------------

len(data[data['income'].astype('str') == '>50K'])

# ---------------------> Number of people having more than 50k salary -----------------

len(data[data['income'].astype('str') == '<=50K'])

# ---------------------> Percentage of people having more than 50k salary -----------------

len(data[data['income'].astype('str') == '<=50K'])

# ---------------------> Preprocessing the dataset ---------------------------------------

income_raw = data['income']
feature_raw = data.iloc[:, :-1]

# ---> Transforming skewed variables using MinMax Scaler

skewed = ['capital-gain', 'capital-loss']
feature_raw[skewed] = feature_raw[skewed].apply(lambda x: np.log(x + 1))
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
feature_raw[numerical] = sc.fit_transform(feature_raw[numerical])

non_num = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
sample = pd.get_dummies(feature_raw[non_num])
feature_raw = pd.concat([feature_raw[numerical], sample], axis = 1)

# ---> Encoding the variables 

income_raw = pd.get_dummies(income_raw, drop_first = True)

# ---------------------> Splitting the data ----------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature_raw, income_raw, shuffle = True, test_size = 0.2, random_state = 0)

# ---------------------> Building the Model ----------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rclassifier = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)
rclassifier.fit(X_train, y_train)

# -----> Evaluating the model performance ----------

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = rclassifier, X = X_train, y = y_train, cv = 10)
print(score.mean())

# -----> Applying Gradient Boost Classifier -------

from sklearn.ensemble import GradientBoostingClassifier
gclassifier = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 1000, random_state = 0)
gclassifier.fit(X_train, y_train)
score = cross_val_score(estimator = gclassifier, X = X_train, y = y_train, cv = 10)
print(score.mean())

# ----------------------> Applying the Test Set transformations --------------------------

data_test = pd.read_csv('test_census.csv')
feature_raw1 = data_test

skewed = ['capital-gain', 'capital-loss']
feature_raw[skewed] = feature_raw[skewed].apply(lambda x: np.log(x + 1))
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

feature_raw_num = data_test[numerical]
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan, strategy = 'median')
feature_raw_num = imputer.fit_transform(feature_raw_num)
feature_raw_num = pd.DataFrame(feature_raw_num, columns = numerical)
sc = MinMaxScaler()
feature_raw_num[numerical] = sc.fit_transform(feature_raw_num[numerical])

non_num = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
feature_raw_cat = data_test[non_num]
feature_raw_cat.fillna(feature_raw_cat.mode().iloc[0], inplace = True)
feature_raw_cat = pd.get_dummies(feature_raw_cat)

feature_raw1 = pd.concat([feature_raw_num, feature_raw_cat], axis = 1)

y_pred = gclassifier.predict(feature_raw1)









