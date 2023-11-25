# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:58:26 2023

@author: Lenovo
"""
#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# VERİ ÖN İŞLEME ------------------------------------------------------------
#veri yükleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

#veri ön işleme
boy = veriler[['boy']]
print(boy)

boykilo = veriler[["boy","kilo"]]
print(boykilo)


#eksik veriler ------------------------------------------------------------

from sklearn.impute import SimpleImputer

#amacımız NaN olan değerleri tamamlamak 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# strategy ne ile impute edeceği yani değiştireceği 
# ortamalama değerini bütün nan değerlere yazacağız

Yas = veriler.iloc[:,1:4].values  # yas kolonunu çekmek için bütün satırlar ve 1. ve 4. kolonları çekiyoruz
print(Yas)

# Yas aslında bir dizi bu dizi üzerinde impute işlemini gerçekleştireceğiz

imputer = imputer.fit(Yas[:,1:4])   

# burada fit() fonksiyonumuz öğrenilecek olan değer.
# fit() fonksiyonu eğitmek için kullanılır.
# Yaşın 1 den 4 e kadar olan kolonlarını öğrenmesini söylüyoruz
# sayısal kolonlar üzerinde imputer öğrenme işlemini yapacak
# simple imputer daki strategy mean olduğu için bu kolonların ortalama değerlerini öğrenecek

# öğrendikten sonra NaN değerleri dönüştürmesini istiyoruz bunun içinde transform fonksiyonunu kullanacağız
# yani fit'le öğretip transformla da öğrendiğini uygulamasını sağlıyoruz
# öğrenilen şey eksik değerlerin yerine konulacak olan değer yani ort değer
# uygulanacak olan şey de nan değerlerin değiştirilmesi
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

#-------------------------------------------------------------------------------------
# kategorik verileri sayısal formata çevirelim
# sadece kategorik verilerle oynayacağımız için kategorik verileri tek başına bir numpy dizisi olarak almaya çalışalım

ulke =veriler.iloc[:,0:1].values  
print(ulke)

# kategorik kolonun dönüşümü için preprocessing altında encoder'ları çağıracağız
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Burada fit() ve transform() fonksiyonlarını beraber çağıracağız
# le.fit_transform dediğimizde veriler iloc daki ilk kolonu alıyor ve bu alınan kolonu transform ediyoruz
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

# ONE-HOT-ENCODİNG
# fit ve transformu tek bir satırda çağırarak tek aşamada ön işlemedeki öğrenme süreci öğrenecek ulke kolonundan
# daha sonra bunu transform edecek
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()  
print(ulke) 

#-----------------------------------------------------------------------------------
# Numpy dizileri dataFrame dönüşümü
# parça parça veri kümelerimiz oluştu nan değerleri giderdik, kategorik verileri one-hat-encoding ile değiştirdiğimiz veri kümesi var
# bu veri parçalarımızı tek bir dataFrame de toplayacağız

# ulke
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ["fr","tr","us"])
print(sonuc)

# yas,boy,kilo
sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ["boy","kilo","yas"])
print(sonuc2)     
                                                    
# cinsiyet kolonunu alalım
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])
print(sonuc3)

# Bu dataframeleri birleştirelim -------------------------------------------
# sonuc + sonuc2 : ilk olarak ulke ve yas,boy,kilo kolonları birleştirildi
s = pd.concat([sonuc,sonuc2], axis = 1)  # axis = 0 kolon düzeyinde, axis = 1 satır düzeyinde birleştirme
print(s)
# s2 = s + sonuc3 : cinsiyet kolonu da sonuc ile birleştirildi
s2 = pd.concat([s,sonuc3], axis = 1)
print(s2)

#----------------------------------------------------------------------------
# Verilerin Eğitim ve Test Olarak Bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)
# s : bağımsız değişkenler yani x
# s3 : bağımlı değişken, sonuç değişkenimiz yani y

#--------------------------------------------------------------------------
# Öznitelik Ölçekleme
# verileri standartlaştırdık

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)























