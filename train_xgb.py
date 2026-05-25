# enconding:utf8
import os
import math
import numpy as np
import pandas as pd
import xgboost as xgb  # 修改点1：导入 xgboost
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 加载训练数据
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0., inplace=True)

label = data_set['label'].values

feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')

training = data_set[feature_list].values

# 加载测试数据
test_data_raw = pickle.load(open('./data/test.pkl','rb'))
test_data_raw.fillna(0., inplace=True)
sub_df = pd.DataFrame(test_data_raw['uid'].copy()) # 确保是DataFrame格式

test_uids = test_data_raw['uid'].values
del test_data_raw['uid']
test_matrix = test_data_raw.values

# 准备测试集的 DMatrix (XGB专用格式)
dtest = xgb.DMatrix(test_matrix, feature_names=feature_list)

kf = KFold(n_splits=5, random_state=2017, shuffle=True)
rmse_list = []
sub_pred = []

print("开始 XGBoost 5折交叉验证训练...")

for train_index, val_index in kf.split(training):
    X_train, y_train = training[train_index], label[train_index]
    X_val, y_val = training[val_index], label[val_index]

    # 修改点2：设置 XGBoost 参数
    xgb_params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror', # 回归任务
        'eta': 0.02,                    # 对应 learning_rate
        'max_depth': 5,                 # 树深
        'subsample': 0.8,               # 对应 bagging_fraction
        'colsample_bytree': 0.8,        # 对应 feature_fraction
        'eval_metric': 'rmse',
        'nthread': 4,
        'verbosity': 0
    }

    # 修改点3：构造 XGBoost 专用的 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_list)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_list)

    # 训练模型
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    # early_stopping_rounds 在 xgb.train 中直接支持
    gbm = xgb.train(xgb_params, dtrain, num_boost_round=1500,
                    evals=watchlist, early_stopping_rounds=100, verbose_eval=50)

    # 验证集预测
    y_pred = gbm.predict(dval)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    print("本折 RMSE:", rmse)
    rmse_list.append(rmse)

    # 测试集预测
    test_pred = gbm.predict(dtest)
    sub_pred.append(test_pred)

print("\nK-Fold RMSE 列表: {}".format(rmse_list))
print("平均 RMSE : {}".format(np.mean(rmse_list)))

# 生成最终结果
pred = np.mean(np.array(sub_pred), axis=0)
sub_df['pred'] = pred

# 保存结果
sub_df.to_csv('submission_xgb.csv', sep=',', header=None, index=False, encoding='utf8')
print("结果已保存至 submission_xgb.csv")

# 修改点4：XGBoost 特征重要性绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(gbm, max_num_features=15, ax=ax)
plt.title("XGBoost 特征重要性排行")
plt.savefig('xgb_importance.png')