# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:18:30 2023

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



# Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10, criterion = 'gini')
# n_estimators : kaç tane ağaç oluşturulacağı
'''
- Criterion parametresi : gini veya entropy kullanılabilmesini sağlayan parametre ,
- Karar ağaçlarında ingormation gain’de bir değerin olasılığı yani bir kolonun oluşması 
ile ilgili olasılık ve bu olasılığın log2 tabanındaki olasılıkla çarpılmasından oluşuyor.
- Gini’nin formülü ise log2 tabanında alınmadan çarpılmasıdır.
'''
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)



# confusing matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('Random forest için confusing matrix \n', cm)









