# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None)
# buradaki csv dosyasının kolon başlıkları olmadığı için yani ilk satırın kolon 
# başlığı olup- olmadığını belirtmek için header = None diyoruz.

t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])
'''
import ettiğimiz kütüphane transaction olarak list of list yani list içinde liste 
istediği için verimizi her bir satırı bir liste olacak şekilde düzenliyoruz bunu da
bir liste içine atıyoruz
'''


from apyori import apriori
# apyori kütüphanesinden apriori'yi import ettik

kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)
# min_lenght : en az 2 ürünü birlikte getirsin. Mesela en az ikili kampanya yapmak istiyoruz

print(list(kurallar))
# kurallar bir obje olduğu için önce liste haline çevirdik