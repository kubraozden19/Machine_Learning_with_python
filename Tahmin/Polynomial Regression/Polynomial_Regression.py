# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:57:13 2023

@author: Lenovo
"""
#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme (slice)
# aldıkları eğitim seviyesinden maaslarını tahmin etmeye çalışacağız
x = veriler.iloc[:,1:2] # x : eğitim seviyesi kolonu
y = veriler.iloc[:,2:]   # y : maas kolonu 

# NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

# Linear Regression  --------------------------------------------------------
# bunlar doğrusal ilişkiye sokulabilecek veriler değil, polinomal bir ilişki içeren veriler
# Ama yine de linear yani doğrusal bir ilişki kursak nasıl olur diye bakalım
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X, Y, color = 'red')
# oluşturulan linear modelin predict fonk. kullanarak her bir x'e karşılık gelen tahminleri görselleştirelim
plt.plot(x, lin_reg.predict(X), color = 'blue')
plt.show()
# fit() ve scatter() da dataFrame problemi olabiliyor onun için bu değerleri 
# örneğin x.values yaparak sadece değerlerini alarak sorunu çözebiliriz


# Polynomial Regressinon -----------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures
# PolynomialFeatures bize herhangi bir sayıyı polinomal olarak ifade etmemize yarıyor
# istediğimiz polinom dereceğini verebiliriz.

poly_reg = PolynomialFeatures(degree = 2)
# linear dünyadaki x değerimi polinomal dünyaya çeviriyoruz
x_poly = poly_reg.fit_transform(X)
print(x_poly)

# x, 1den 10a kadar giden sayılardan oluştuğu için kareler de 1den 10a kadar giden sayılardan oluşacak

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
# x_poly ile oluşturulmuş olan değişkenleri (x^0, x^1, x^2'yi) kullanarak y'yi öğren 
# yani x^0, x^1, x^2'yi al beta(0), beta(1) ve beta(2) öğren buradaki her bir katsayı farklı

plt.scatter(X, Y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#-----------------------------------------------------------------------------
# Mesela Regression Boyutunu 4 yapalım, 4. dereceden bir regression oluşturalım  
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.show()


# Tahminler -----------------------------------------------------------------

# Linear Regression
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))


# Polynomial Regression
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))







