# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 01:03:33 2023

@author: Lenovo
"""
#kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yükleme
veriler  = pd.read_csv('veriler.csv')
print(veriler)

# KATEGORİK VERİLERİ DÖNÜŞTÜRELİM---------------------------------------------
# regresyon algoritmaları matematiksel denklemler üzerinden çalışıyor. Yani doğrunun denklemini çizebilmemiz için 
# 3 boyutlu uzayda noktalar oluşturmamız gerekiyor. ve bu değerlerde sayısal değerlerden oluşuyor
# cinsiyet kolonunu dönüştürelim encoding kullanarak kategorik -> numeric

cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# LabelEncoder ile fit işlemi, kategorik bir değişkenin sınıflarını öğrenip her bir sınıfa bir sayı atar.
#  öğrenilen parametreler kullanılarak veri kümesi üzerinde dönüşüm işlemi gerçekleştirilir. 
# Bu aşamada, öğrenilen parametrelerin (örneğin, etiketlerin sayısal karşılıkları) veri kümesine uygulanmasıyla dönüşüm gerçekleştirilir.
cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(cinsiyet)


ohe = preprocessing.OneHotEncoder()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

# ulke kategorik verisi için de aynı şeyi yapalım -------------
ulke = veriler.iloc[:,0:1].values
print(ulke)

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#--------------------------------------------------------------------------------------------------
# Numpy dizileri DataFrame dönüşümü
df1 = pd.DataFrame(data = ulke, index = range(22), columns = ["fr","tr","us"])
print(df1)

# Daha önce dummy variable (kukla değişken) söz etmiştik. Bu arada kategorik verileri sayısal verilere dönüştürdük evet 
# ama bunları veri setimize eklerken dikkat etmeliydik örneğin cinsiyet kolonunun one-hot-encoding sonucu 2 kolona dönüşmesi sonucu
# eğer bu 2 kolonu da veri setime eklersem bir risk oluşturur çünkü zaten aslında bu 2 kolon birbirini ifade edebilen kolonlar 
# hatta buna dummy variable trap (kukla değişken tuzağı) demiştik. Birbirini içerebilen bu 2 kolonu almak risk içerir.
# Bu nedenle bu kolonlardan sadece bir tanesini almamız yeterli, 
# aslında burada şunu anlıyoruz cinsiyet kolonu için sadece label encoder işlemi yeterliymiş ohe işlemine gerek yokmuş :) 
df2 = pd.DataFrame(data = cinsiyet[:,:1], index = range(22), columns = ["cinsiyet"])
print(df2)

# ilk başta zaten sayısal olarak gelen verileri de alalım
sayısal_veriler = veriler.iloc[:,1:4].values
df3 = pd.DataFrame(data = sayısal_veriler, index = range(22), columns = ["boy","kilo","yas"])
print(df3)


#------------------------------------------------------------------------------------------
# DataFrameleri birlestirelim

sonuc1 = pd.concat([df1, df3], axis = 1)
print(sonuc1)

sonuc2 = pd.concat([df2,sonuc1], axis = 1)
print(sonuc2)




#-------------------------------------------------------------------------------------
# Verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonuc1, df2, test_size = 0.33, random_state = 0)
# sonuc1 : x değerleri yani bağımsız değişkenler
# df3 : y değeri, bağımlı değişken yani sonuc değeri

# Multiple Linear Regression 

from sklearn.linear_model import LinearRegression
# simple linear reg : bir tane bağımsız değişkenimiz vardı
# multiple linear reg : birçok bağımsız değişkenimiz var

regressor = LinearRegression()
model = regressor.fit(x_train, y_train)

# burada öğrenilen doğru birden fazla boyuttan oluşacak,
# yani bu örnek veride 6 bağımsız değişken olduğu için doğrumuz 6 boyuttan oluşacak

tahmin = model.predict(x_test)

#------------------------------------------------------------------------------
# Mesela boy tahmini için Multiple Linear Regression kullanalım
# Bunun için veri setinden "boy" kolonunu çekmemiz gerekiyor çünkü bu kolon artık bizim y değerimiz olacak
# Yani artık bizim regresyon sonucunda tahmin etmek istediğimiz değer y kolonu

boy = sonuc2.iloc[:,4:5].values
print(boy)

solda_kalan_kolonlar = sonuc2.iloc[:,0:4]
print(solda_kalan_kolonlar)
sagda_kalan_kolonlar = sonuc2.iloc[:,5:]
print(sagda_kalan_kolonlar)

veri = pd.concat([solda_kalan_kolonlar, sagda_kalan_kolonlar], axis = 1)
print(veri)

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size = 0.33, random_state = 0)
# boy : y değişkeni yani sonuc değeri, bağımlı değişken
# veri : x değerleri yani bağımsız değişkenler

regressor = LinearRegression()
model = regressor.fit(x_train, y_train)

tahmin = model.predict(x_test)

#-------------------------------------------------------------------------------
# BACKWARD ELEMINATION (Geri Eleme Yöntemi)
# Bağımsız değişken seçimi için kullanılan yöntemlerden biri olan Geri Eleme Yöntemini inceleyelim

import statsmodels.api as sm
# modelin ve modeldeki değişkenlerin başarısıyla ilgili bir sistem kurabiliriz.

# İlk olarak değişkenlerimin üzerinde bir model oluşturmak için değişkenleri sisteme eklemeliyiz
# bir dizi oluşturacağız bu dizi içerisine bütün değişkenleri koyacağız,
# sonrasında sırasıyla değişkenleri eleyerek ilerleyeceğiz, hangi değişken sistemi daha fazla bozuyorsa 
# hangi değişkenin p-value'su daha yüksekse, sl'yi (significance level) geçiyorsa onu sistemden çıkaracağız

# Linear Regressiondaki matematiksel modeli hatırlarsak beta gibi bir sabit değerimiz vardı,
# yani bütün değişkenlerin bir beta çarpanı vardı ve hata miktarı vardı ve bir de beta(0) diye bir sabit değer vardı
# Bu sabit değer şuan bizim sistemimizde bulunmuyor. Şuan ki verilere baktığımızda 6 tane kolon var 
# ama bu kolonlardan herhangi birisi sabit değeri ifade etmiyor. 
# Bu 6 kolonu birden sisteme dahil edecek olursak sabit değer sistemde bulunmayacak
# dolayısıyla her değişken için her satır için bir sabit değişken olsun diye buraya bir dizi ekleyeceğiz
# bu dizi 22 satırdan oluştuğu için buraya 22 tane 1 ekliyoruz bunun nedeni çarpanı 1 olması

X = np.append( arr = np.ones((22,1)).astype(int), values = veri, axis = 1)
# veri'ye 22x1 boyutlarında 1'lerden oluşan ve veri tipi int olan bir matris eklendi
# sisteme beta(0) değerlerini yani sabit değeri eklemiş olduk

# veri DataFrameindeki her bir kolonu ifade edecek bir liste oluşturacağız ve bu liste üzerinde eleme yaparak ilerleyeceğiz
X_l = veri.iloc[:,[0,1,2,3,4,5]].values 
#bütün kolonları aldık hepsinin p-value değerini hesaplayacağız ve eleme yaparken buradan bazı kolonları çıkaracağız

X_l = np.array(X_l, dtype = float)

# istatisitksel modelimizi çıkarmaya yarıyor sm.OLS()
# boy kolonunu bulmak istediğimiz için yani boy kolonu etiket olduğu için onu belirttik
# X_l 'de bağlantı kurmasını istediğimiz değerler, bağımsız değişkenleri içeren dizi
model = sm.OLS(boy, X_l).fit()
# Yani her bir kolonun, bağımsız değişkenin boy kolonu üzerindeki etkisini ölçüyorki buna göre model başarısını analiz edebilelim
print(model.summary())




X_l = veri.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())



