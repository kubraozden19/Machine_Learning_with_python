# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:03:26 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB
N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
toplam_odul = 0
secilenler = []
birler = [0] * d  # her bir ilan için birlerin tıklanma sayısı
sifirlar = [0] * d  # her bir ilan için sıfırların tıklanma sayısı

for n in range(1,N):
    # her bir ilanı tıklanıp tıklanmadığına bakacağız eğer tıklandıysa bu tıklama değerini döndüreceğiz
    ad = 0 #seçilen ilan
    max_th = 0
    
    for i in range(0,d): # bütün ilanların tek tek ihtimallerine bakacağız
        rasbeta = random.betavariate (birler[i] + 1, sifirlar[i] + 1)
  
        if rasbeta > max_th :
            max_th = rasbeta
            ad = i
            
            
        secilenler.append(ad)
        odul = veriler.values[n,ad]  
        
        if odul == 1:
            birler[ad] = birler[ad] + 1
        else:
            sifirlar[ad] = sifirlar[ad] + 1
        
        toplam_odul = toplam_odul + odul

print('Toplam Ödül :')    
print(toplam_odul)

plt.hist(secilenler)
plt.show()

'''
birler ve sifirlar diye 2 tane dizi ekledik bunlar ilk başta sıfırdan başladı bunlar üzerinden bir beta dağılımı
aldık bu beta dağılımıın en yüksek olduğu değeri buluyoruz o ana kadar ki ilanı aklımızda tutuyoruz. Ve birler ve 
sıfırları da arttırıyoruz. Thompson reinforced learning kullanıyor rastgele değil gözleme göre kendisini geliştiren 

'''