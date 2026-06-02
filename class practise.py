
#  IMPORTNING THE LIBRARY

import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd		

#--------------------------------------------

# import the dataset & divided my dataset into independe & dependent

dataset = pd.read_csv(r"D:\Data Science\DS_prakash_senapati\Notes prakash sir\april(2025)-1\april(2025)\18th- ML\18th- ML\5. Data preprocessing\Data.csv")

X = dataset.iloc[:, :-1].values	

y = dataset.iloc[:,3].values  

#--------------------------------------------
## missing value tretment  by transformer:-

from sklearn.impute import SimpleImputer # SPYDER 4  

# sklearn.impute is a Scikit-learn module used to handle missing data
# SimpleImputer = used to handle missing values (NaN) in a dataset.
# SimpleImputer, which replaces missing values using strategies such as mean, median, most frequent value, 
# or a constant value before training machine learning models.

imputer = SimpleImputer( ) # is called parameter tunning its take by default strategy is mode 

# imputer = SimpleImputer(strategy="median")  # is call hyperparameter tunning 
# is call hyperparameter tunning we use strategy median and mode apply in numerical column only


# imputation - in independent variable x --number

imputer = imputer.fit(X[:,1:3]) 

# fit() calculates and learns statistics from the data (mean, median, etc.).

X[:, 1:3] = imputer.transform(X[:,1:3])

# transform() applies those learned values to replace missing data.
# fit_transform() performs both operations together.
# parameter tunning - i follow as per system parameter



#-----------------------------------------------------------------------------

#  HOW TO ENCODE CATEGORICAL DATA & CREATE A DUMMY VARIABLE

# # imputation - dependent and independent variable into  labelencoder (means NO-0 AND YES-1)

from sklearn.preprocessing import LabelEncoder

# sklearn.preprocessing is a Scikit-learn module used to transform raw data into a format suitable for machine learning algorithms. 
# It provides tools for encoding categorical variables, feature scaling, normalization, and data transformation.

# LabelEncoder- this transformaer is used to convert categorical text values into numerical values.

# independent variable -x
labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0]) 

X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

#-------------------------------------------------------------------------------

 #  dependent variable - y  

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#-----------------------------------------------------------------------

#SPLITING THE DATASET IN TRAINING SET & TESTING SET

from sklearn.model_selection import train_test_split

# The sklearn. model_selection module in Scikit-Learn provides functions for splitting data into training and test sets, evaluating machine learning models, and performing cross-validation. 
# train_test_split() is used to split a dataset into training data and testing data.

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state=0) 

# test_size=0.2 mean - It means 20% of the data is used for testing and 80% is used for training.
# if you remove random_stat then your model not behave as accurate 
# random_state controls the random shuffling of data before splitting. Using the same value ensures reproducible results.

#-----------------------------------------------------------------------

#FEATURE SCALING

from sklearn.preprocessing import Normalizer

sc_X = Normalizer() 

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

#---------------------------------------------------------------------













