#! /usr/bin/python3
# xgboost-based models grid search / cross_validation / predict

import csv
import pandas as pd
import scipy
import xgboost as xgb
import numpy as np
from datetime import datetime
from math import fsum, fabs, sqrt, floor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def mse(preds, dtrain):
	labels = dtrain.get_label()
	return 'mse', fsum((labels - preds) ** 2) / len(labels)

def grid_search(param, data, search_param):
	best_score = 10000000
	best_param = {}
	
	names = search_param.keys()
	print("search params: ", end = '')
	for name in names:
		print(name, ' ', end = '')
	print()
	
	param_list = []
	for name in names:
		vals = search_param[name]
		param_list = [[x] for x in vals] if len(param_list) == 0 else [x + [y] for x in param_list for y in vals]

	for p in param_list:
		for i, name in enumerate(names):
			param[name] = p[i]
		ret = xgb.cv(param, data, num_boost_round = 6000, nfold = 5, early_stopping_rounds = 100,
				feval = mse, verbose_eval = False)
		score = float(ret.tail(1).to_string().split()[9])
		r = int(ret.tail(1).to_string().split()[8])
		if score < best_score:
			best_score = score
			best_param = p
		print('Testing param:', p, ' score:', score, 'at round', r)	
	
	print('Total parameter combination:', len(param_list))
	print("Best param: ")
	for i, name in enumerate(names):
		print('    ', name, ' = ', best_param[i], sep='')
	print("Best score:", best_score)
	
# manual cross validation (ignore)
def cross_validation(param, data, nfold):
	score_mean = 0
	iter_mean = 0
	npf = floor(data.num_row() / nfold)	
	for i in range(nfold):
		dval = data.slice(list(range(i * npf, (i + 1) * npf)))
		dtrain = data.slice(list(range(i * npf)) + list(range((i + 1) * npf, data.num_row())))
		
		bst = xgb.train(param, dtrain, num_boost_round = 6000, early_stopping_rounds = 100, 
				evals = [(dval, 'val')],  feval = mse,  verbose_eval = False)
		score_mean += bst.best_score
		iter_mean += bst.best_iteration
		print('fold', i, '( score:', bst.best_score, ', iteration:', bst.best_iteration, ')')	
	
	score_mean /= nfold
	iter_mean /= nfold
	print('  mean_score:', score_mean, ' mean_iteration:', iter_mean)
	return score_mean	


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

# set large weights for M
weight = [1 if x >= 6 else 1 for x in train['血糖']]

dtrain = xgb.DMatrix(train[train.columns.drop('血糖')].values, label = train['血糖'].values, weight = weight)
dtest = xgb.DMatrix(test.values)

	
# set params for gbtree / gblinear
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
					
param_gblinear = {
		'booster': 'gblinear',
		'eta': 	1,
		'lambda': 0.001,
		'alpha': 20,
		'lambda_bias': 0.001,		
		'objective': 'reg:linear',
		'base_score':  base_score,
		'silent': 1}
			
param_list = {'gbtree': param_gbtree, 'gblinear': param_gblinear}
task_list = {0: 'grid search', 1: 'cv', 2: 'train and test'}
# set task id
task = 1
	
# set model name
model = 'gbtree'
param = param_list[model]
	
if task_list[task] == 'grid search':
	print("**********  grid search  **********")	
	search_param = {'eta': [0.001, 0.005, 0.01 ,0.05, 1, 5, 10], 'alpha': [0.1, 0.5, 1, 2,5, 10, 15, 20]}
	grid_search(param, dtrain, search_param)
if task_list[task] == 'cv':
	print("**********  cross validation  **********")	
	ret = xgb.cv(param, dtrain, num_boost_round = 6000, nfold = 5, 
		early_stopping_rounds = 100, feval = mse, verbose_eval = False)
	print(ret)	
	#cross_validation(param, dtrain, nfold = 5)
if task_list[task] == 'train and test':
	print("**********  train and predict  **********")		
	best_iteration = {'gbtree': 260, 'gblinear': 1616}
	boost_num = best_iteration[model] # should be decided by the cv
	bst = xgb.train(param, dtrain, num_boost_round = boost_num, feval = mse)
	pred = bst.predict(dtest)
	pred = [[x] for x in pred]
	dt = datetime.now()
	output = dt.strftime('%y%m%d_%H%M.csv')
	with open(output, 'w', newline = '') as pred_file:
		pred_writer = csv.writer(pred_file, delimiter = ',')
		pred_writer.writerows(pred)
		print("Save predictions of testset to", output)
	pred = bst.predict(dtrain)
	print('mse of trainset:', mean_squared_error(pred, dtrain.get_label()))		
	pred = [[x] for x in pred]
	dt = datetime.now()
	output = dt.strftime('train_pred_%y%m%d_%H%M.csv')
	#with open(output, 'w', newline = '') as pred_file:
		#pred_writer = csv.writer(pred_file, delimiter = ',')
		#pred_writer.writerows(pred)
		#print("Save predictions of trainset to", output)			


