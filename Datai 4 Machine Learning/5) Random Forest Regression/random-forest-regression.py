# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:09:38 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv(r"C:\Users\Hamza\Desktop\Datai 4 Machine Learning\5) Random Forest Regression\randomforestreg.csv",sep=";", header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)   #100 decision tree, 42 subdata, can be any number
rf.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu: ",rf.predict([[7.8]]))

y_head=rf.predict(x)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)  #her değerdeki y_pred karşılığını görebilmek için
y_head_ = rf.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head_,color="green")   #decision treeden daha iyi çalıştı.100 tree reg is better than 1
plt.xlabel("tribun level") 
plt.ylabel("ucret")
plt.show()
from sklearn.metrics import r2_score
print("r score:",r2_score(y,y_head))