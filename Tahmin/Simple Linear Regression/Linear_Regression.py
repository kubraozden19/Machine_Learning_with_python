# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:08:29 2023

@author: Lenovo
"""
#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv('satislar.csv')
print(veriler)

# veri ön işleme
aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

# benzer olarak şu şekilde de yapılabilir
aylar1 = veriler.iloc[:,0:1].values
print(aylar1)


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state = 0)


''' VERİLERİ ÖLCEKLEMEK İSTERSEK
# verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

Y_train = scaler.fit_transform(y_train)
Y_test = scaler.fit_transform(y_test)
'''




# LINEAR REGRESSION MODELİ OLUŞTURALIM-----------------------------------------------------------

from sklearn.linear_model import LinearRegression
# LinearRegression classını eklemiş olduk şimdi bu classdan obje oluşturalım

lr = LinearRegression()

# X_train ve Y_train bilgilerini alarak bir model inşaa et diyoruz
lr.fit(x_train, y_train)


# modeli uygulayalım
# tahmin etmesini istediğimiz şey X_test, yani bunu tahmin etsin ve gerçek değer olan Y_test ile karşılaştıralım
tahmin = lr.predict(x_test)

# VERİ MODELİMİZİ GÖRSELLEŞTİRELİM-------------------------------------------------
# verileri çizmeden önce sıralamamız gerekiyor çünkü biz verilerimizi seçerken random olarak seçmiştik
# yani mesela ilk 5. veriyi çizdiriyor sonra 3. veriyi gibi bu şekilde olduğunda saçma sapan bir görsel çıkıyor
# Bu yüzden verilerimizi görselleştirmeden önce sıralamamız gerekiyor.

x_train = x_train.sort_index()
# y_train'ini de sıralamalıyız yoksa sıralanan x_train'deki her bir satıra y_train'deki başka bir ayın karşılığı geliyor
# sıraladığımızda x_train'deki doğru karşılığa gelecek çünkü indexe göre sıralıyoruz verilere göre sıralamıyoruz
# eğer verilere göre sıralasaydık o zaman en küçük ay en küçük satış değeri ile eşleşecekti ve bu hataya neden olur
y_train = y_train.sort_index()  

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")
