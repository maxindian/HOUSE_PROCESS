# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
import evaluation as ev
import numpy as np
num_n = 1400
path = 'E:/mycode/Saleprice/'
train = pd.read_csv(path+'train.csv')

# test = pd.read_csv(path+'test.csv')

def train_test_split(data):
	msk = np.random.rand(len(data)) < 0.8
	train = data[msk]
	test = data[~msk]
	return (train,test)
feature_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','TotalBsmtSF','TotRmsAbvGrd','GarageArea','YrSold']

train,test = train_test_split(train)

# x_train = train[feature_cols]

train = train.fillna(0)
test = test.fillna(0)
# x_train = train[feature_cols].head(num_n)
# x_train = x_train.fillna(0)
# y_train = train.SalePrice.head(num_n)
lm = LinearRegression()
lm.fit(train[feature_cols],train['SalePrice'])
#预测误差
# x_test = train[feature_cols].loc[num_n:]
# x_test = x_test.fillna(0)
pr_y = lm.predict(test[feature_cols])
pr_y = pd.core.series.Series(pr_y)

y = test['SalePrice'].reset_index(drop = True)


print(y.ix[5:10])
exit()
print(np.sqrt((y - pr_y)**2).mean() / y.mean())
# print(y)
# print(pr_y)
exit()

trainfs_y = train.SalePrice.loc[num_n:]
print(ev.evaluation(pr_y,trainfs_y,num_n))

x_test2 = test[feature_cols]
x_test2 = x_test.fillna(0)
y_pred = lm.predict(x_test2)
ev.input(y_pred)
# 线性回归