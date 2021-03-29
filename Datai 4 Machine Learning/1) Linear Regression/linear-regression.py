# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:07:28 2018

@author: user
"""
# import library
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("dataset.csv",sep = ";")

# plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% linear regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)   #reshape veriyi (14,1) hale getirmek için
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)   #linear regression içine x,y fit ettik

#%% prediction
import numpy as np


b0_ = linear_reg.intercept_   #♦find b0 value
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept

b1 = linear_reg.coef_      # y = b0+b1x
print("b1: ",b1)   # egim slope

# maas = 1663 + 1138*deneyim 

maas_yeni = 1663 + 1138*11
print(maas_yeni)       #beklenen maaş

print(linear_reg.predict([[11]]))   #maaşı tahmin et

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim


plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(x)  # maas

plt.plot(x, y_head,color = "red")

linear_reg.predict([[100]])

from sklearn.metrics import r2_score

print('r score:',r2_score(y,y_head))










