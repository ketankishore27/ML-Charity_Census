# ---------------------> Importing the required libraries ------------------------------

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
#from sklearn.ensemble import RandomForestClassifier
#rclassifier = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)
#rclassifier.fit(X_train, y_train)
#
## ---------> Preidcting the results --------
#
#y_pred1 = rclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#from sklearn.metrics import confusion_matrix
#cm1 = confusion_matrix(y_test, y_pred1)
#
## -----> Evaluating the model performance ----------
#
#from sklearn.model_selection import cross_val_score
#score = cross_val_score(estimator = rclassifier, X = X_train, y = y_train, cv = 10)
#print(score.mean())
#
## -----> Applying Gradient Boost Classifier -------
#
#from sklearn.ensemble import GradientBoostingClassifier
#gclassifier = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 1000, random_state = 0)
#gclassifier.fit(X_train, y_train)
#score = cross_val_score(estimator = gclassifier, X = X_train, y = y_train, cv = 10)
#print(score.mean())
#
## ---------> Preidcting the results --------
#
#y_pred2 = gclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#from sklearn.metrics import confusion_matrix
#cm2 = confusion_matrix(y_test, y_pred2)

# ----------------------> Applying the Test Set transformations --------------------------
#
#data_test = pd.read_csv('test_census.csv')
#feature_raw1 = data_test
#
#skewed = ['capital-gain', 'capital-loss']
#feature_raw[skewed] = feature_raw[skewed].apply(lambda x: np.log(x + 1))
#numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
#
#feature_raw_num = data_test[numerical]
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = np.nan, strategy = 'median')
#feature_raw_num = imputer.fit_transform(feature_raw_num)
#feature_raw_num = pd.DataFrame(feature_raw_num, columns = numerical)
#sc = MinMaxScaler()
#feature_raw_num[numerical] = sc.fit_transform(feature_raw_num[numerical])
#
#non_num = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
#feature_raw_cat = data_test[non_num]
#feature_raw_cat.fillna(feature_raw_cat.mode().iloc[0], inplace = True)
#feature_raw_cat = pd.get_dummies(feature_raw_cat)
#
#feature_raw1 = pd.concat([feature_raw_num, feature_raw_cat], axis = 1)
#
#y_pred = gclassifier.predict(feature_raw1)

# -------------------> Trying Support Vector Alogrithm ---------------------------------

#from sklearn.svm import SVC
#sclassifier = SVC(kernel = 'rbf', random_state = 0)
#sclassifier.fit(X_train, y_train)
#
## ---------> Preidcting the results --------
#
#y_pred3 = sclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#cm3 = confusion_matrix(y_test, y_pred3)

# ------------------> Trying Naive Base Classifier ------------------------------------

#from sklearn.naive_bayes import GaussianNB
#gnclassifier = GaussianNB()
#gnclassifier.fit(X_train, y_train)
#
## ---------> Preidcting the results --------
#
#y_pred5 = gnclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#cm5 = confusion_matrix(y_test, y_pred5)
#
## ------------------> Trying Neural Networks ----------------------------------------
#
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#
#nclassifier = Sequential()
#
## ---> Adding 1st layers --
#
#nclassifier.add(Dense(output_dim = 52, init = 'uniform', activation = 'relu', input_dim = 103))
#
## ---> Adding Second Layer --
#
#nclassifier.add(Dense(output_dim = 52, activation = 'relu', init = 'uniform'))
#
## ---> Adding third input layer --
#
#nclassifier.add(Dense(output_dim = 52, activation = 'relu', init = 'uniform'))
#
## ----> Adding fourth input layer --
#
#nclassifier.add(Dense(output_dim = 52, activation = 'relu', init = 'uniform'))
#
## ----> Adding final layer ---------
#
#nclassifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#
## ----> Compiling the model
#
#nclassifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
## ----> Fitting the data ---------
#
#nclassifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
#
## ---------> Preidcting the results --------
#
#y_pred6 = rclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#cm6 = confusion_matrix(y_test, y_pred6)
#
## ---------> Trying XGBoost Algorithm ------------------------------------------------
#
#from xgboost import XGBClassifier
#xclassifier = XGBClassifier()
#xclassifier.fit(X_train, y_train)
#
## ---------> Preidcting the results --------
#
#y_pred7 = rclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#cm7 = confusion_matrix(y_test, y_pred7)
#
## ---------> Trying KNN Classifier --------------------------------------------------
#
#from sklearn.neighbors import KNeighborsClassifier
#kclassifier = KNeighborsClassifier(n_neighbors = 100)
#kclassifier.fit(X_train, y_train)
#
## ---------> Preidcting the results --------
#
#y_pred8 = rclassifier.predict(X_test)
#
## ---------> Checking the confusion matrix--
#
#cm8 = confusion_matrix(y_test, y_pred8)


# ------------------> Trying Adaboost Algorithm ---------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
aclassifier = AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(random_state = 0, criterion='entropy'), 
    n_estimators = 100, learning_rate = 0.1)
aclassifier.fit(X_train, y_train)

# ---------> Preidcting the results --------

y_pred = aclassifier.predict(X_test)

# ---------> Checking the confusion matrix--

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
ascore = accuracy_score(y_test, y_pred)
scorer = make_scorer(fbeta_score, beta = 0.5)

cm = confusion_matrix(y_test, y_pred)

# ----> Finding the best parameters ------

parameters = [{'n_estimators': [50, 100, 150],
               'learning_rate': [0.1, 0.5, 1],
               'base_estimator__min_samples_split': np.arange(2, 8, 2),
               'base_estimator__max_depth': np.arange(1, 4, 1)}]

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator = aclassifier,
                    param_grid = parameters, scoring = scorer, verbose = 5, cv = 10, n_jobs = -1)
grid.fit(X_train, y_train)
    
aclassifier = grid.best_estimator_

y_pred = aclassifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
score = cross_val_score(grid.best_estimator_, X = X_train, y = y_train, cv = 10, n_jobs = -1)
print(score.mean())

# --------------------> Finding the important features ------------------------------

important = aclassifier.feature_importances_

# --------------------> Cloning the Estimator ---------------------------------------

from sklearn.base import clone
oclassifier = clone(aclassifier)

X_train_reduced = X_train.iloc[:, np.argsort(important)[::-1][:10]]
X_test_reduced = X_test.iloc[:, np.argsort(important)[::-1][:10]]

oclassifier.fit(X_train_reduced, y_train)

y_pred1 = oclassifier.predict(X_test_reduced)

cm1 = confusion_matrix(y_test, y_pred)

 ----------------------> Applying the Test Set transformations --------------------------

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

y_pred = aclassifier.predict(feature_raw1)


