# encoding:utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# ====================== 加载 UCI 数据集 ======================
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"]).values
y = df["default payment next month"].values

# ====================== 消融实验 4种方案 ======================
ablation_methods = [
    ("基础LightGBM（无采样）", None),
    ("LightGBM + SMOTE", SMOTE(random_state=2017)),
    ("LightGBM + ENN", EditedNearestNeighbours()),
    ("LightGBM + SMOTEENN（本文）", SMOTEENN(random_state=2017)),
]

skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)

print("=" * 70)
print("             🧪 消融实验：不同不平衡处理方法对比")
print("=" * 70)

for name, sampler in ablation_methods:
    auc_list, recall_list, f1_list = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 采样
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        params = {
            'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
            'max_depth': 5, 'num_leaves': 21, 'learning_rate': 0.02,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'verbose': -1
        }

        model = lgb.train(
            params, lgb.Dataset(X_train, y_train),
            num_boost_round=1500, valid_sets=[lgb.Dataset(X_val, y_val)],
            callbacks=[lgb.early_stopping(100)]
        )

        y_pred = model.predict(X_val)
        y_bin = (y_pred >= 0.5).astype(int)

        auc_list.append(roc_auc_score(y_val, y_pred))
        recall_list.append(recall_score(y_val, y_bin))
        f1_list.append(f1_score(y_val, y_bin))

    print(f"【{name}】")
    print(f"  AUC    : {np.mean(auc_list):.5f}")
    print(f"  召回率  : {np.mean(recall_list):.5f}")
    print(f"  F1     : {np.mean(f1_list):.5f}")
    print("-" * 70)