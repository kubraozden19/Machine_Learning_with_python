# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:37:46 2023

@author: Lenovo
"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# NumPy dizi (array) dönüşümü
X = x.values
Y = y.values


# Decision Tree Regression ----------------------------------------------------
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state = 0)

dt_reg.fit(X,Y) # Y:maas, X:pozisyonlar
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color = 'black')
plt.plot(X, dt_reg.predict(X), color = 'yellow')

plt.plot(x, dt_reg.predict(Z), color = 'green')
plt.plot(x, dt_reg.predict(K), color = 'yellow')

plt.show()
print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))

from sklearn.metrics import r2_score
print("Decision Tree R2 değeri")
print(r2_score(Y, dt_reg.predict(X)))




# Random Forest (Rassal Ağaclar) Regression -----------------------------------

from sklearn.ensemble import RandomForestRegressor
# Tahmin için kullanılan algoritmalar genelde regeressor şeklinde geçiyor

rf_reg = RandomForestRegressor(n_estimators = 10,random_state = 0)
# n_estimators : estimators, tahmin demek bunun anlamı kaç tane decision tree çizileceği
# random forest algoritmamız burada 10 tane farklı decision tree çizecek verinin 10 farklı parçasıyle

rf_reg.fit(X, Y.ravel()) # öğrenme işlemini gerçekleştiriyoruz
print(rf_reg.predict([[6.5]])) # tahmin 

plt.scatter(X, Y, color = 'red')
plt.plot(X, rf_reg.predict(X), color = 'blue')

plt.plot(X, rf_reg.predict(Z), color = 'green')
plt.plot(X, rf_reg.predict(K), color = 'yellow')


'''
Decision Tree de ağacımızdaki veriler tamamen net veriler üzerindendi, Decision tree öğrenme aşamasındaki 
veriler dışında bir veri döndüremezdi. Ama random forest da birden fazla decision tree oluşturuyorduk ve 
bu birden fazla decision tree neticede ortalamalar döndürebiliyordu yani 2 tane farklı decision tree nin 
farklı görüşünü alıp bu görüşlere göre ortak bir değer döndürüyor. Dolayısıyla buradaki orjinal veriler 
dışında veriler dönmesi random forest da mümkün olabiliyor tahminde, sınıflandırma da decision tree de 
random forest da farklı bişey döndüremez
'''

# R2 Hesaplanması---------------------------------------------------------
from sklearn.metrics import r2_score

print("Random Forest R2 değeri")
print(r2_score(Y, rf_reg.predict(X)))
# Y : gerçek değer
# rf_reg.predict(X) : tahmin edilen değer

print("K için :",r2_score(Y, rf_reg.predict(K)))
print("Z için :",r2_score(Y, rf_reg.predict(Z)))






