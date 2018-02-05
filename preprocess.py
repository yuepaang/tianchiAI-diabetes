#! /usr/bin/python3

import copy
import scipy
import numpy as np
import math
from sklearn.preprocessing import Imputer, StandardScaler

def preprocess_data(train, test):
	train = remove_attrs(train, ['id', '体检日期'])
	train = use_one_hot(train, ['性别'])
	train = to_nparray(train[1:], '')
	
	print((np.isnan(train) == False).sum(0))

	
	print('train shape:', train.shape)
	over_attr_thresh = ((np.isnan(train) == False).sum(0) >= 1500)
	train = np.array([[x for i, x in enumerate(row) if over_attr_thresh[i] == True] for row in train], dtype = float)
	over_samp_thresh = ((np.isnan(train) == False).sum(1) >= 15)
	train = np.array([row for i, row in enumerate(train) if over_samp_thresh[i] == True], dtype = float)	
	print('remove sparse r/c:', train.shape)
	
	test = remove_attrs(test, ['id', '体检日期'])
	test = use_one_hot(test, ['性别'])
	test = to_nparray(test[1:], '')
	
	print('test shape:', test.shape)
	test = np.array([[x for i, x in enumerate(row) if over_attr_thresh[i] == True] for row in test], dtype = float)
	print('remove sparse r/c:', test.shape)
	
	ntrain = train.shape[0]
	ntest = test.shape[0]
	con = np.concatenate((train, test), axis = 0)
	mask = np.isnan(con)
	
	imp = Imputer(copy = False, missing_values = 'NaN', strategy = 'mean', axis = 0)
	imp.fit_transform(con)

	sts = StandardScaler(copy = False, with_mean = True, with_std = True)
	sts.fit_transform(con)	

	#con[mask] = np.nan
	train = to_csr(con[:ntrain])
	test = to_csr(con[ntrain:])
	
	return train, test
	
def preprocess_label(label):
	res = [float(x) for x in label[1:]]
	return res

def remove_attrs(data, attrs):
	inds = [data[0].index(attr) for attr in attrs if attr in data[0]]
	res = [[x for i, x in enumerate(row) if i not in inds] for row in data]
	return res
		
def use_one_hot(data, attrs):
	res = copy.deepcopy(data)
	for attr in attrs:
		if attr not in res[0]:
			continue
		ind = res[0].index(attr)
		vals = set([row[ind] for row in data[1:] if row[ind] != ''])
		for i, row in enumerate(res):
			val = row.pop(ind)
			for j, x in enumerate(vals):
				if i == 0:
					row.insert(ind + j, attr + '_' + str(x))
				else:
					row.insert(ind + j, (1 if val == x else 0) if val in vals else '')	
	return res

def remove_sparse_attrs(data, thresh):
	over_thresh = ((np.isnan(data) == False).sum(0) >= thresh)
	res = np.array([[x for i, x in enumerate(row) if over_thresh[i] == True] for row in data], dtype = float)
	return res

def remove_sparse_samps(data, thresh):

	return res

def to_nparray(data, miss):
	return np.array([[float(x) if x != miss else np.nan for x in row] for row in data], dtype = float)

def to_csr(data):
	row_inds = [i for i, row in enumerate(data) for x in row if x != np.nan]
	col_inds = [i for row in data for i, x in enumerate(row) if x != np.nan]
	vals = [float(x) for row in data for x in row if x != np.nan]
	return scipy.sparse.csr_matrix((vals, (row_inds, col_inds)))

	
# samples of missing attr are removed
def split_by_ordinal(train, test, attr):
	missing = ['', '??']
	attrs = train[0]
	if attr not in attrs:
		return ([train], [test])
	ind = attrs.index(attr)
	vals = set([row[ind] for row in train[1:] if row[ind] not in missing])
	vals = [val for val in vals]
	vals.sort()
	
	attr_row = train[0][:ind] + train[0][ind + 1:] 
	trains, tests = [], []
	for val in vals:
		trains.append([row[:ind] + row[ind + 1:] for row in train if row[ind] == attr or row[ind] == val])
		tests.append([row[:ind] + row[ind + 1:] for row in test if row[ind] == attr or row[ind] == val])
	
	return (trains, tests)
