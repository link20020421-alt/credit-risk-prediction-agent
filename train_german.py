# encoding:utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# ====================== 加载 德国信贷数据集 ======================
df = pd.read_csv("data/german.data-numeric", header=None, sep="\s+")

# 最后一列是标签：1=好(0)，2=坏(1) → 转换成 0/1
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == 2, 1, 0)  # 标签转换

feature_list = [f"f{i}" for i in range(X.shape[1])]

# ====================== 5 折 + SMOTEENN ======================
skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)
auc_list, recall_list, f1_list = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 不平衡处理
    smote = SMOTEENN(random_state=2017)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 5,
        'num_leaves': 21,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1
    }

    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val)

    model = lgb.train(
        params, train_set,
        num_boost_round=1500,
        valid_sets=val_set,
        callbacks=[lgb.early_stopping(100)]
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_bin = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred_bin)
    f1 = f1_score(y_val, y_pred_bin)

    auc_list.append(auc)
    recall_list.append(recall)
    f1_list.append(f1)

# ====================== 输出 德国数据集结果 ======================
print("\n" + "="*50)
print("           德国信贷数据集 5 折结果")
print("="*50)
print(f"AUC    : {np.mean(auc_list):.5f}")
print(f"召回率  : {np.mean(recall_list):.5f}")
print(f"F1     : {np.mean(f1_list):.5f}")
print("="*50)