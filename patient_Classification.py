import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_curve
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import  make_classification


a=[]
data =pd.read_csv('./sample/14patient_feature.txt',sep='\t',header=None)

data1 = data[1]

for i in data1:
	if i==1:#1
		a.append(1)
	else:
		j = 0
		a.append(j)
#data2 = data.drop([0,1281],axis=1)
data2 = data.drop([0,1],axis=1)

X_train,valid_x,y_train,valid_y =train_test_split(data2,a,test_size=0.2,random_state=0)
# 创建成lgb特征的数据集格式
train = lgb.Dataset(X_train, y_train) # 将数据保存到LightGBM二进制文件将使加载更快
valid = lgb.Dataset(valid_x, valid_y, reference=train)  # 创建验证数据

parameters = {
					  'num_leaves': [20,25,30,35,40,45,50,55,60,65],
					  'max_depth': [-1, 1, 2,3,4,5,7,9,10,15,17,25],#-1
					  'learning_rate': [0.01,0.03,0.05,0.06,0.07,0.2,0.1,0.3,0.5],
					  'feature_fraction': [0.5,0.6, 0.7,0.8, 0.9,0.95,1],
					  'bagging_fraction': [0.5,0.6, 0.7,0.8, 0.9,0.95,1],
					  'bagging_freq': [2, 3,4, 5, 6,7, 8,9,10],
					  'lambda_l1': [0, 0.4, 0.5,0.6,0.7,0.8,0.9],
					  'lambda_l2': [0, 10,20,30, 40,50],
					  'cat_smooth': [1, 10, 15, 20,30, 35,50],#1




}
gbm = lgb.LGBMClassifier(boosting_type='gbdt' ,  #gbdt,dart
						 objective = 'binary' ,
						 metric = 'auc' ,
						 verbose = 0,
						 max_depth=5,  # 9(5类5 20 0.1 1 0.8 2 0 0)
						 num_leaves = 20,  #  25
						 learning_rate = 0.1 ,  #0.1
						 bagging_fraction= 1,  #0.8
						 feature_fraction=0.8,  #0.5
						 bagging_freq=2,  #2
						 lambda_l1= 0,#0
						 lambda_l2= 0,#10
						 n_jobs=10
						 )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='roc_auc', cv=3)#n_jobs=1/cpu使用个数设置
gsearch.fit(X_train, y_train)
p=gsearch.predict_proba(valid_x)
fpr,tpr,threshold = roc_curve(valid_y,p[:,1].ravel())

print("Best score: %0.2f" % gsearch.best_score_)
print("最优参：")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))
print("......................................")

