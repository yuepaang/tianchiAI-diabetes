#! /bin/usr/python3
# linear models in sklearn

import csv
import numpy as np
from math import floor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import ensemble

train = '../data/d_train_20180102.csv'
test = '../data/d_test_A_20180102.csv'

def process_data(train, test):
	print('initial train shape (with label):', train.shape)
	print('initial test shape:', test.shape)	
	train = train.drop(['id'], axis = 1)
	test = test.drop(['id'], axis = 1)
	train['性别'] = train['性别'].map({'男':1, '女':0})
	test['性别'] = test['性别'].map({'男':1, '女':0})
	train['体检日期'] = train['体检日期'].map(lambda x: int(x[0:2]) + int(x[3:5]) * 100)
	test['体检日期'] = test['体检日期'].map(lambda x: int(x[0:2]) + int(x[3:5]) * 100)
	
	col_filter = (pd.notna(train).sum() >= 0)
	train = train.loc[:, col_filter]
	train = train.loc[train['血糖'] < 20, :]
	
	test = test.loc[:, col_filter.drop('血糖')]

	train.fillna(train.median(),  inplace = True)
	test.fillna(test.median(), inplace = True)	
	
	sta_scaler = preprocessing.StandardScaler().fit(train.drop(['血糖'], axis = 1))
	train[train.columns[:-1]] = sta_scaler.transform(train.drop(['血糖'], axis = 1))
	test[test.columns] = sta_scaler.transform(test)
	
	print('train shape (with label):', train.shape)
	print('test shape:', test.shape)		
	
	return train, test

train_path = '../data/d_train_20180102.csv'
test_path = '../data/d_test_A_20180102.csv'

train_df = pd.read_csv(train_path, encoding = 'gb18030')
test_df = pd.read_csv(test_path, encoding = 'gb18030')

print("********** preprocess data **********")	
train, test = process_data(train_df, test_df)
	
task_list = {0: 'grid search', 1: 'cv', 2: 'train and test'}
# set task id
task = 1
model = 'rf'
print('model used:', model)
if task_list[task] == 'grid search':
	print('********** grid search **********')
	if model == 'ridge':
		ridge = linear_model.RidgeCV(alphas = np.logspace(-7, 3, 20), cv = 5)
		ridge.fit(train.drop(['血糖'], axis = 1).values, train['血糖'].values)
		print('best alpha for ridge:', ridge.alpha_)
	if model == 'lasso':	
		lasso = linear_model.LassoCV(alphas = np.logspace(-7, 3, 20), cv = 5, max_iter = 1000000)
		lasso.fit(train.drop(['血糖'], axis = 1).values, train['血糖'].values)
		print('best alpha for lasso:', lasso.alpha_)
if task_list[task] == 'cv':
	print('********** cross validation **********')
	score_mean = 0
	n_splits = 5
	kf = KFold(n_splits = n_splits)
	for i, (train_ind, val_ind) in enumerate(kf.split(train)):
		tra = train.drop(['血糖'], axis = 1).values[train_ind]
		tra_label = train['血糖'].values[train_ind]
		val = train.drop(['血糖'], axis = 1).values[val_ind]
		val_label = train['血糖'].values[val_ind]
		pred = []
		if model == 'ridge':
			ridge = linear_model.Ridge(alpha = 88.58)
			ridge.fit(tra, tra_label)
			pred = ridge.predict(val)
		if model == 'lasso':
			lasso = linear_model.Lasso(alpha = 0.005455)	
			lasso.fit(tra, tra_label)
			pred = lasso.predict(val)
		if model == 'knn':
			knn = neighbors.KNeighborsRegressor(n_neighbors = 25, weights = 'uniform', n_jobs = -1)
			knn.fit(tra, tra_label)
			pred = knn.predict(val)
		if model == 'svm':
			svr = svm.SVR(kernel='rbf', C = 10, gamma = 0.01)
			svr.fit(tra, tra_label)
			pred = svr.predict(val)
		if model == 'bg_knn':
			knn = neighbors.KNeighborsRegressor(n_neighbors = 25, weights = 'uniform', n_jobs = -1)
			bg_knn = ensemble.BaggingRegressor(knn, n_estimators = 40, max_samples = 0.6, max_features = 0.8, n_jobs = -1)
			bg_knn.fit(tra, tra_label)
			pred = bg_knn.predict(val)
		if model == 'rf':
			rf = ensemble.RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', n_jobs = -1)
			rf.fit(tra, tra_label)
			pred = rf.predict(val)
				
		score = mean_squared_error(pred, val_label)
		score_mean += score
		print('fold', i, '(score):', score)
	score_mean /= n_splits
	print('  mean_score:', score_mean)
	
if task_list[task] == 'train and test':
	ridge.fit(dtrain, train_label)
	pred = ridge.predict(dval)			
	
