import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from datetime import datetime

with open('../d_train_20180102.csv') as f:
    train_df = pd.read_csv(f)
with open('../d_test_A_20180102.csv') as f:
    test_df = pd.read_csv(f)

train_y = train_df.pop('血糖')
cols = train_df.columns[[1, 2, 4, 5, 6, 7, 12, 13, 15,
    16, 18, 24, 25 ,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
# Ludan's data
train_df_lu = train_df[cols]
test_df_lu = test_df[cols]

# convert gender
def g2d(x):
    if x == '男':
        return 1
    else:
        return 0
# Yue's data
lg0 = list(map(g2d, train_df['性别'].tolist()))
train_df['性别'] = lg0
lg1 = list(map(g2d, test_df['性别'].tolist()))
test_df['性别'] = lg1
ld0 = list(map(lambda x:int(x.replace('/', '')), train_df['体检日期'].tolist()))
train_df['体检日期'] = ld0
ld1 = list(map(lambda x:int(x.replace('/', '')), test_df['体检日期'].tolist()))
test_df['体检日期'] = ld1
train_df.drop(train_df.columns[[2,18, 19, 20, 21, 22]], axis=1, inplace=True)
test_df.drop(test_df.columns[[2, 18, 19, 20, 21, 22]], axis=1, inplace=True)
train_df = train_df.fillna(train_df.median(skipna=True))
test_df = test_df.fillna(test_df.median(skipna=True))

# Ludan's
lg0 = list(map(g2d, train_df_lu['性别'].tolist()))
train_df_lu['性别'] = lg0
lg1 = list(map(g2d, test_df_lu['性别'].tolist()))
test_df_lu['性别'] = lg1
# XGBoost Model
model_xgb0 = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.9, gamma=1, learning_rate=0.07, max_delta_step=0,
       max_depth=7, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1.0)
# model_xgb0 = xgb.XGBRegressor(max_depth=6, learning_rate=0.01, n_estimators=800, min_child_weight=3.1,
#                                subsample=0.6, colsample_bytree=0.7, gamma=3.7, reg_alpha=5,
#                                 reg_lambda=1.2, scale_pos_weight=1,
#                                max_delta_step=0, colsample_bylevel=1, eval_metric='rmse', random_state=0)
model_xgb0.fit(train_df_lu, train_y)


# Lasso Model
lassocv = LassoCV(alphas=np.logspace(-10, 10, 500), cv=5, normalize=True)
lassocv.fit(train_df, train_y)
model_lasso = Lasso(alpha=lassocv.alpha_, normalize=True)
model_lasso.fit(train_df, train_y)
magic_lasso = pd.read_csv('../Experiment/lasso.csv', header=None)
pred19 = 0.6*model_xgb0.predict(test_df_lu) + 0.4*model_lasso.predict(test_df)

best = pd.read_csv('../best.csv', header=None)
print(mean_squared_error(best[0].tolist(), pred19)/2)