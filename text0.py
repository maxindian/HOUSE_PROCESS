import pandas as pd
import matplotlib.pyplot as plt
from numpy import * 
import numpy as np
from sklearn import linear_model
from scipy.stats import norm, skew #for some statistics
from scipy import stats
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
train = pd.read_csv('E:/mycode/Saleprice/train.csv',quoting = 3).fillna(0)
test = pd.read_csv("E:/mycode/Saleprice/test.csv",quoting = 3).fillna(0)

def data_processing(data):

	numeric_feats = data.dtypes[data.dtypes != "object"].index  # 把内容为数值的特征找出来
	xArr = data[numeric_feats[0:-1]]
	yArr = data[numeric_feats[-1]].T
	return xArr,yArr

train_x,train_y = data_processing(train)

numeric_feats = test.dtypes[test.dtypes != "object"].index  # 把内容为数值的特征找出来
test_x = test[numeric_feats[:]]


model = linear_model.LogisticRegression()
model.fit(train_x,train_y)
y_pred = model.predict(test_x)
