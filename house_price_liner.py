# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
import evaluation as ev
import numpy as np
path = 'E:/mycode/Saleprice/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
feature_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','TotalBsmtSF','TotRmsAbvGrd','GarageArea','YrSold']
x_train = train[feature_cols].head(1300)
x_train = x_train.fillna(0)
y_train = train.SalePrice.head(1300)
lm = LinearRegression()
lm.fit(x_train,y_train)
#预测误差
x_test = train[feature_cols].loc[1300:]
x_test = x_test.fillna(0)
pr_y = lm.predict(x_test)
pr_y = pd.core.series.Series(pr_y)
trainfs_y = train.SalePrice.loc[1300:]
#trainfs_y = np.ndarray(trainfs_y)
#(pr_y[0]-trainfs_y[1000])
#print(pr_y)

#print("============================")
#print(trainfs_y[1000])
#print(trainfs_y)
#print(ev.evaluation(pr_y,trainfs_y))
x = 34279.76219512475
y = 180921.1959
print((x/y))


#x_test = test[feature_cols]
#x_test = x_test.fillna(0)
#y_pred = lm.predict(x_test)


# 线性回归