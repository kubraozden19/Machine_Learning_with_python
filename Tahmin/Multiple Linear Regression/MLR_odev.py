# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:21:58 2023

@author: Lenovo
"""

#kütüphaneler 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#verilerin yüklenmesi
veriler = pd.read_csv("odev_tenis.csv")
print(veriler)

# Veri Ön İşleme
# Kategorik verilerin dönüştürülmesi

play = veriler.iloc[:,-1:].values
print(play)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# play kategorik değişkenini dönüştürelim
play[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(play)
# play bağımsız değişkeni binomnal olduğu için one-hot-encoder yapmamıza gerek yok


# windy kategorik değişkenini dönüştürelim,
# bu değişken boolen değerler aldığı için direkt astype(int) ile 0-1 'e dönüştürebiliriz
windy = veriler.iloc[:,3].astype(int)
print(windy)
# windy bağımsız değişkeni binomnal olduğu için one-hot-encoder yapmamıza gerek yok


# outlook kategorik verisini dönüştürelim
outlook = veriler.iloc[:,:1].values
outlook[:,0]= le.fit_transform(veriler.iloc[:,0])
print(outlook)

# outlook kolonu için one-hot-encoder yapalım. Çünkü polinomial :)
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

#-----------------------------------------------------------------------------
# Numpy dizilerini DataFrame dönüştürme

df1 = pd.DataFrame(data = outlook, index = range(14), columns = ["overcast","sunny","rainy"])

df2 = pd.DataFrame(data = windy, index = range(14), columns = ["windy"])

df3 = pd.DataFrame(data = play, index = range(14), columns = ["play"])

temperature = veriler.iloc[:,1:2].values
print(temperature)
df4 = pd.DataFrame(data = temperature, index = range(14), columns = ["temperature"])

humidity = veriler.iloc[:,2:3].values
print(humidity)
df5 = pd.DataFrame(data = humidity, index = range(14), columns = ["humidity"])

#-----------------------------------------------------------------------------
# DataFramelerin birleştirilmesi

s1 = pd.concat([df1, df2], axis = 1)
print(s1)
s2 = pd.concat([df3, df4], axis = 1)
print(s2)

veri = pd.concat([s1,s2], axis = 1)
print(veri)


#-----------------------------------------------------------------------------
# verilerin Eğitim ve Test için bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(veri, df5, test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
model = regressor.fit(x_train, y_train)

tahmin = model.predict(x_test)

#----------------------------------------------------------------------------
# BACKWARD ELEMINATION (Geri Eleme Yöntemi)
# Bağımsız değişken seçimi için kullanılan yöntemlerden biri olan Geri Eleme Yöntemini inceleyelim

import statsmodels.api as sm

# linear regresyon için sabit değer ekleyelim
X = np.append(arr = np.ones((14,1)).astype(int), values = veri, axis = 1 )

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(humidity, X_l).fit()
print(model.summary())


# OLS Regression Resulta göre x4 bağımsız değişkeninin p-value değeri  0.593 bu nedenle bunu çıkaralım
X_l = veri.iloc[:,[0,1,2,4,5]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(humidity, X_l).fit()
print(model.summary())



