# encoding:utf8
"""补充运行 Borderline-SMOTE 和 KMeans-SMOTE，与 sampling_compare.py 完全一致的实验设置"""
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# 加载 JDD 数据（路径和你 sampling_compare.py 一样）
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file, 'rb'))
data_set.fillna(0., inplace=True)

label = np.where(data_set['label'].values >= 0.5, 1, 0).astype(int)
feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')
X = data_set[feature_list].values
y = label

# 参数和你 sampling_compare.py 完全一致
RANDOM_STATE = 2017
cv = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
lgb_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
    'max_depth': 5, 'num_leaves': 21, 'learning_rate': 0.02,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'verbose': -1, 'random_state': RANDOM_STATE
}
base_model = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)

# 只跑两个缺少的方法
methods = {
    "Borderline-SMOTE": BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5),
    "KMeans-SMOTE": KMeansSMOTE(random_state=RANDOM_STATE, k_neighbors=5),
}

print("=" * 70)
print("  补充实验：Borderline-SMOTE / KMeans-SMOTE (JDD数据集)")
print("=" * 70)
print(f"{'采样方法':<25}{'AUC':<12}{'召回率':<12}{'F1值':<12}")
print("-" * 70)

for method_name, sampler in methods.items():
    aucs, recalls, f1s = [], [], []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        model = base_model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        y_bin = (y_pred >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    print(f"{method_name:<25}{np.mean(aucs):<12.4f}{np.mean(recalls):<12.4f}{np.mean(f1s):<12.4f}")

print("-" * 70)
print("跑完后把上面两行数字发给我，我填进论文表5。")
