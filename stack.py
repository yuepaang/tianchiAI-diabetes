#! /bin/usr/python3
# stack multiple models

import csv
import numpy as np
from math import floor
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from datetime import datetime

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

train_path = '../d_train_20180102.csv'
test_path = '../d_test_A_20180102.csv'

train_df = pd.read_csv(train_path, encoding = 'gb18030')
test_df = pd.read_csv(test_path, encoding = 'gb18030')

print("********** preprocess data **********")	
train, test = process_data(train_df, test_df)
	
task_list = {0: 'cv', 1: 'train and test'}
# set task id
task = 1

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
		
		pred = pd.DataFrame()
		# lasso
		lasso = linear_model.Lasso(alpha = 0.005455)	
		lasso.fit(tra, tra_label)
		pred['lasso'] = lasso.predict(val)
		# knn
		knn = neighbors.KNeighborsRegressor(n_neighbors = 25, weights = 'uniform', n_jobs = -1)
		knn.fit(tra, tra_label)
		pred['knn'] = knn.predict(val)
		# svr
		svr = svm.SVR(kernel='rbf', C = 10, gamma = 0.01)
		svr.fit(tra, tra_label)
		pred['svr'] = svr.predict(val)
		# xgboost
		dtrain = xgb.DMatrix(tra, label = tra_label)
		dval = xgb.DMatrix(val)		
		base_score = train['血糖'].sum() / train['血糖'].shape[0]
		param_gbtree = {
			'booster': 'gbtree',
			'eta': 0.01,
			'gamma': 0,
			'max_depth': 5,
			'min_child_weight': 1,
			'max_delta_step': 0,
			'subsample': 0.6,
			'colsample_bytree': 0.8,
			'colsample_bylevel': 1,
			'lambda': 8,
			'alpha': 0,
			'objective': 'reg:linear',
			'base_score': base_score,
			'silent': 1}		
		bst = xgb.train(param_gbtree, dtrain, num_boost_round = 1100)
		pred['xgboost'] = bst.predict(dval)			
		# rf
		rf = ensemble.RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', n_jobs = -1)
		rf.fit(tra, tra_label)
		#pred['rf'] = rf.predict(val)			
		
		pred = [x[3] if x.mean() > 6 else x.mean() for x in pred.values]
			
		score = mean_squared_error(pred, val_label)
		score_mean += score
		print('fold', i, '(score):', score)
	score_mean /= n_splits
	print('  mean_score:', score_mean)
	
if task_list[task] == 'train and test':
	tra = train.drop(['血糖'], axis = 1).values
	tra_label = train['血糖'].values
	tes = test.values	
	
	pred = pd.DataFrame()
	# lasso
	lasso = linear_model.Lasso(alpha = 0.005455)	
	lasso.fit(tra, tra_label)
	pred['lasso'] = lasso.predict(tes)
	# knn
	knn = neighbors.KNeighborsRegressor(n_neighbors = 25, weights = 'uniform', n_jobs = -1)
	knn.fit(tra, tra_label)
	pred['knn'] = knn.predict(tes)
	# svr
	svr = svm.SVR(kernel='rbf', C = 10, gamma = 0.01)
	svr.fit(tra, tra_label)
	pred['svr'] = svr.predict(tes)
	# xgboost
	dtrain = xgb.DMatrix(tra, label = tra_label)
	dtest = xgb.DMatrix(tes)		
	base_score = train['血糖'].sum() / train['血糖'].shape[0]
	param_gbtree = {
			'booster': 'gbtree',
			'eta': 0.01,
			'gamma': 0,
			'max_depth': 5,
			'min_child_weight': 1,
			'max_delta_step': 0,
			'subsample': 0.6,
			'colsample_bytree': 0.8,
			'colsample_bylevel': 1,
			'lambda': 8,
			'alpha': 0,
			'objective': 'reg:linear',
			'base_score': base_score,
			'silent': 1}		
	bst = xgb.train(param_gbtree, dtrain, num_boost_round = 1100)
	pred['xgboost'] = bst.predict(dtest)	
	# rf
	rf = ensemble.RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', n_jobs = -1)
	rf.fit(tra, tra_label)
	#pred['rf'] = rf.predict(tes)	
	
	pred = [x[3] if x.mean() > 6 else x.mean() for x in pred.values]	
	pred = [[x] for x in pred]
	dt = datetime.now()
	output = dt.strftime('%y%m%d_%H%M.csv')
	with open(output, 'w', newline = '') as pred_file:
		pred_writer = csv.writer(pred_file, delimiter = ',')
		pred_writer.writerows(pred)
		print("Save predictions of testset to", output)		
	
