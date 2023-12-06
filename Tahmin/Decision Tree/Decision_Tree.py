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
plt.show() # plt.show demezsek daha önceki plotun üzerine çizmeye devam eder dolayısıyla 2 plotu üst üste görürüz


# tahmin
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))




# Decision Tree ---------------------------------------------------------
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






