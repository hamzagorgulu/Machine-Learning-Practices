# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:15:14 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision.csv",sep = ";",header = None)   #başlıkta data olmaması içini

x = df.iloc[:,0].values.reshape(-1,1)   #0. column x
y = df.iloc[:,1].values.reshape(-1,1)   #1. column y

#%%  decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()   # random sate = 0
tree_reg.fit(x,y)


tree_reg.predict([[5.5]])
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)   #min xten max xe kadar .01 aralıkla değer oluştur
y_head = tree_reg.predict(x_)
# %% visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()