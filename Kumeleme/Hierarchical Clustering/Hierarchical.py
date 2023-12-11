# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:58:49 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:]

# Hierarchical Clustering---------------------------------------------
from sklearn.cluster import AgglomerativeClustering

h_ac = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
#n_clustering : kaç küme olacağı
#affinity (yakınlık) : doğrusallık demek yani mesafeler arasındaki farka bakacak
#linkage (bağlantı) : 2 küme arasındaki mesafeyi hangi yöntem ile bulacağı

y_tahmin = h_ac.fit_predict(X)
# fit() sistemi öğreniyor inşa ediyor.fit_predict():hem inşaa ediyor hem tahmin ediyor

print("tahmin sonucu",y_tahmin)

'''
y_tahmin sonucu ile her bir veri noktasının (3 küme olsun istediğimiz için)
 0,1,2 şeklinde hangi kümede olduğunu gösterdi.
'''

plt.scatter(X[y_tahmin == 0].iloc[:,0],X[y_tahmin == 0].iloc[:,1], s=100, c='red')
'''
X[y_tahmin == 0]: Bu ifade, y_tahmin dizisindeki değeri 0 olan örneklerin bulunduğu satırları seçer. 
Bu, veri kümesindeki yalnızca küme 0'a ait olan satırları içeren bir DataFrame'i temsil eder.
iloc[:, 0]: Bu ifade, yukarıdaki DataFrame'in sadece ilk sütununu seçer. Yani, küme 0'a ait olan 
örneklerin x-koordinatlarını temsil eder. iloc[:, 1]: Benzer şekilde, bu ifade ikinci sütunu seçer 
ve küme 0'a ait olan örneklerin y-koordinatlarını temsil eder.
s=100: Bu, scatter plot üzerindeki her bir noktanın boyutunu belirler. Burada noktaların boyutu 100 
olarak belirlenmiştir. c='red': Bu, scatter plot üzerindeki noktaların rengini belirler.
Küme 0'a ait noktalar burada kırmızı renkte gösterilir. Bu kod satırı, küme 0'a ait örnekleri gösteren 
bir scatter plot oluşturur ve bu verileri yalnızca iloc metodu kullanılarak indekslenen DataFrame içinden seçer.
'''
# diğer 2 cluster içinde aynı şekilde 
plt.scatter(X[y_tahmin == 1].iloc[:,0], X[y_tahmin == 1].iloc[:,1], s=100, c='blue')
plt.scatter(X[y_tahmin == 2].iloc[:,0], X[y_tahmin == 2].iloc[:,1], s=100, c='green')


# Dendrogram-------------------------------------------------------------------------
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
#sch.linkage(X, method='ward'): Bu kısım, veri setiniz üzerinde "ward" bağlantı yöntemini kullanarak bir bağlantı matrisi oluşturur.
plt.show()













