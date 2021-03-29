# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:11:43 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polyreg.csv",sep = ";")   #böyle ayırınca başlıkları da aldı

x = df.araba_max_hiz.values.reshape(-1,1)
y = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

# linear regression =  y = b0 + b1*x
# multiple linear regression   y = b0 + b1*x1 + b2*x2

# %% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head2 = lr.predict(x)  #price prediction

plt.plot(x,y_head2,color="red",label ="linear")  #iyi bir model değil
plt.show()



# %%
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)   #2. dereceden denklem old belirtmek için

x_polynomial = polynomial_regression.fit_transform(x)  # x i polinoma uygun hale çevirdik. fit_transform


# %% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

# %%

y_head2 = linear_regression2.predict(x_polynomial)   #polinomlu hale göre predict ediyor

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()





















