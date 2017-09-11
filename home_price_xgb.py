#coding=utf-8
import numpy as np
import matplotlib as plt
import pandas as pd
from scipy.stats import skew
import xgboost as xgb
import evaluation as ev
num_n = 1400 
train = pd.read_csv('train.csv')
trainfs_y = train.SalePrice.loc[num_n:]
test = pd.read_csv('test.csv')
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
price = pd.DataFrame({"price":train["SalePrice"],"log(price+1)":np.log1p(train["SalePrice"])})
#print(price.hist()) 取后5行
train['SalePrice'] = np.log1p(train['SalePrice'])

#将偏斜度大于0.75的数值列log转换，使之尽量符合正态分布
numeric_feats = all_data.dtypes[all_data.dtypes !='object'].index
skewed_feats = train[numeric_feats].apply(lambda x:skew(x.dropna()))#计算偏斜
skewed_feats = skewed_feats[skewed_feats>0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#将字符串特征列中的内容分别提出来作为新的特征出现，表现为0、1
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_text = all_data[train.shape[0]:]
y = train.SalePrice 
X = X_train
dtrain = xgb.DMatrix(X_train,label = y)
dtest = xgb.DMatrix(X_text)
params = {"max_depth":2,"eta":0.1}
model = xgb.cv(params, dtrain , num_boost_round=500,early_stopping_rounds = 100)
model_xgb = xgb.XGBRegressor(n_estimators = 360,max_depth = 2,learning_rate = 0.1)
model_xgb.fit(X_train,y)
#preds = model_xgb.predict(X_text)
pr_y = np.exp(model_xgb.predict(X[num_n:]))
pr_y = pd.core.series.Series(pr_y)
print(ev.evaluation(pr_y,trainfs_y,num_n))