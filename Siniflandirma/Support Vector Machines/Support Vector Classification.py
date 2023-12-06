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



# Support Vector Machine
from sklearn.svm import SVC

svc = SVC(kernel = 'linear')
'''
kernel, destek vektör makinelerinde (SVM - Support Vector Machine) kullanılan bir parametredir 
ve SVM'in veri uzayında karar sınırlarını belirlemek için nasıl çalışacağını kontrol eder.
SVM, doğrusal ve doğrusal olmayan karar sınırları oluşturabilir. kernel parametresi, 
bu sınırları belirlerken kullanılacak matematiksel fonksiyonu ifade eder.
'''
svc.fit(X_train, y_train.ravel())
# modelimiz doğrusal bir ayrım noktası bulmaya çalışacak

y_pred = svc.predict(X_test)



# Confusing Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('SVC için confusing matrix \n', cm)

'''
başarıyı değiştirmek için svc içerisinde kernel'ı değiştirebiliriz. rbf, poly, sigmoid gibi
'''



