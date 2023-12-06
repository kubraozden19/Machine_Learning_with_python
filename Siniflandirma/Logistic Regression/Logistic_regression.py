# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:00:11 2023

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

# Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train.ravel())

y_pred = log_reg.predict(X_test)
print(y_pred)
print(y_test)




# Confusing Matrix-------------------------------------------------
from sklearn.metrics import confusion_matrix

#confusing matris aynı verinin gerçek değeri ve tahmin değeri üzerinde oluşturulur.
cm = confusion_matrix(y_test, y_pred)
# y_test : gerçek değer
# y_pred : tahmin değer

print("confusing matris "),
print(cm)




