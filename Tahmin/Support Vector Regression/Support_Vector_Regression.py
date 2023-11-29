# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 00:58:04 2023

@author: Lenovo
"""
# Önemli noktalardan biri support vector regression'ın scaler ile kullanılma zorunluluğu
# yani SVR veriler üzerinde bir model oluştururken autlier veriler veya marjinal diyebileceğimiz
# çok aşırı aykırı olan verilere karşı hassasiyeti var buna karşı bir dayanıklılığı yok bu algoritmanın
# dolayısıyla SVR kullanırken dikkat etmemiz gereken konularden birisi scalerı kullanmak !!


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


#linear regression
#doğrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#polynomial regression
#doğrusal olmayan (non-linear) model olusturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# 4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


#Görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()



#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))




# verilerin ölçeklenmesi

from sklearn.preprocessing import StandardScaler
 
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)
 

# SVR --------------------------------------------------------------------

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
# SVR() ın default kernel parametresi 'rbf' olarak tanımlı, istersek bunu polynomial ya da lineaar yapabiliriz

svr_reg.fit(x_olcekli, y_olcekli) # ölceklediğimiz x ve y değerleri arasında bağlantıyı kurabilmesi için fit() ettik

plt.scatter(x_olcekli, y_olcekli, color = "green") 
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color = "red")
 
# tahmin
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))






