# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 03:08:48 2023

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


# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski')
# n_neighbors : kaç komşuya bakılacağı
# metric : mesafe ölçümünde kullanılacak metric

'''
n_neighbors : kaç komşuya bakılacağı
bu sayıyı çok yüksek verirsek sanki çok iyi çalışır ve ne kadar çok komuşuya bakarsa o kadar 
daha iyi sınıflandırır gibi düşünüyoruz ama olay öyle değil !!

'''

knn.fit(x_train, y_train.ravel()) # eğitim
#ravel() kullanarak y'yi sütun vektörüne çevirdik 
y_pred = knn.predict(X_test)



# Confusing Metrics
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('confusing matrix  \n',cm)




