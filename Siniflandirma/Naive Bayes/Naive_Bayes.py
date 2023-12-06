# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:36:39 2023

@author: Lenovo
"""

#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


# verilerin eğitim-test olarak ayrılması
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)


# verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



# Naive Bayes -----------------------------------------------
# Burada Gaussian Naive Bayes'i kullandık
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train.ravel())

y_pred = gnb.predict(X_test)



# Confusing Matris 
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('Gaussian Naive Bayes için Confusing Matrix \n', cm)











