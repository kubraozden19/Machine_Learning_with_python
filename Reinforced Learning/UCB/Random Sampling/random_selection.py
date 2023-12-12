
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random 

N = 10000 #10000 tane rastgele sayı üretmek için
d = 10  # seçilecek seçenek sayısı
toplam_odul = 0
secilenler = []

for n in range(0, N):
    ad = random.randrange(d) # 10 ilandan birisi için random sayı (1-10 arası)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # satır sayısı ve ilan seçimi , verilerdeki n. satır = 1 ise odul 1, değil ise odul 0 
    toplam_odul = toplam_odul + odul
    
    
plt.plot(secilenler)
plt.show()
    
    
    









