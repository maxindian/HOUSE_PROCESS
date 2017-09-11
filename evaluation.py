import pandas as pd
import numpy as np
import math
def evaluation(pr_y,trainfs_y,num_n):
	sum_a = 0.0
	n = pr_y.shape[0]
	for i in range(n):
		sum_a += (pr_y[i] - trainfs_y[i+num_n]) ** 2
	wucha = (math.sqrt(sum_a/n)/180921.1959)
	return wucha

def input(y_pred):
	test['SalePrice'] = y_pred
	test[['Id','SalePrice']].to_csv('submission1.csv',index = None)	