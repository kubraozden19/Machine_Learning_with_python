# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:24:59 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:]

# K-Means -------------------------------------------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
# n_cluster : kaç kümeye ayrılacağı
# init : 'k-means++' : merkez noktaların başlangıç değerlerini belirlemek için 'k-means++' yöntemini kullanır

kmeans.fit(X) # eğitim

print(kmeans.cluster_centers_) # clusterlardaki merkez noktaların hacim ve maas bilgilerini gösterir.





# K için optimum değeri bulmaya çalışalım

sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    '''
    Buradaki amaç k-means algoritmasının her seferinde yeniden aynı random değer ile aynı başlangıç
    değer ile k-means++ algoritmasını kullanarak farklı bir cluster sayısını denemek 
    '''
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    # kmeans.inertia_ : bu k-means'in WCSS değerleri, yani k-means'in ne kadar başarılı olduğu
    
    
plt.plot(range(1,11), sonuclar)  # k,1'den 10'a kadar her çalıştırmadaki WCSS değerlerini çizdirelim
    
    
    
    
    
    
    
    



