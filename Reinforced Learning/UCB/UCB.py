# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:32:38 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB
N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
# Ri(n)
oduller = [0] * d # ilk basta bütün ilanların değeri sıfır
# Ni(n)
tiklamalar = [0] * d # o ana kadarki tıklamalar 
toplam_odul = 0
secilenler = []

for n in range(1,N):
    # her bir ilanı tıklanıp tıklanmadığına bakacağız eğer tıklandıysa bu tıklama değerini döndüreceğiz
    ad = 0 #seçilen ilan
    max_ucb = 0
    
    for i in range(0,d): # bütün ilanların tek tek ihtimallerine bakacağız
  
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
            ucb = ortalama + delta # i. ilan için ucb
        else: 
            ucb = N*10
        if max_ucb < ucb : # max'tan büyük bir ucb çıktı 
            max_ucb = ucb
            ad = i
            
        secilenler.append(ad)
        tiklamalar[ad] = tiklamalar[ad] + 1 # bir ilana tıkladıysak onun tıklama değerini arttıracağız
        odul = veriler.values[n,ad]  
        oduller[ad] =  oduller[ad] + odul  # o ilanın odulunu de  veya  arttır. 
        toplam_odul = toplam_odul + odul

print('Toplam Ödül :')    
print(toplam_odul)

plt.hist(secilenler)
plt.show()


'''
Burada 10000 tane ilanı dönüyoruz. içinde de her bir ilan tıklması eylemi için her bir satır için 
hangi ilana tıklayacağımı bulmaya yarayan bir döndü daha dönüyorum bu döngünün amacı o ilanın teker teker 
değerlerine bak bunların içinde en fazla UCB değerine sahip olanı bul.
'''


