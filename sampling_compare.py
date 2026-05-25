# encoding:utf8
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 加载数据（和你train.py路径完全一致，不用改）======================
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file, 'rb'))
data_set.fillna(0., inplace=True)

label = np.where(data_set['label'].values >= 0.5, 1, 0).astype(int)
feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')
X = data_set[feature_list].values
y = label

# ====================== 2. 实验配置（和你论文里的参数完全一致）======================
# 固定随机种子，保证结果可复现
RANDOM_STATE = 2017
# 5折交叉验证，和你论文保持一致
cv = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
# LightGBM参数，和你train.py完全一致
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 5,
    'num_leaves': 21,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': -1,
    'random_state': RANDOM_STATE
}
base_model = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)

# ====================== 3. 待对比的主流采样方法 ======================
sampling_method_list = {
    "无采样(基线LightGBM)": None,
    "随机欠采样": RandomUnderSampler(random_state=RANDOM_STATE),
    "SMOTE过采样": SMOTE(random_state=RANDOM_STATE),
    "ADASYN过采样": ADASYN(random_state=RANDOM_STATE),
    "SMOTETomek混合采样": SMOTETomek(random_state=RANDOM_STATE),
    "SMOTEENN混合采样(本文方法)": SMOTEENN(random_state=RANDOM_STATE)
}

# ====================== 4. 5折交叉验证跑对比实验 ======================
print("\n" + "=" * 90)
print("【论文用 不同采样方法性能对比结果】")
print("=" * 90)
print(f"{'采样方法':<30}{'AUC':<12}{'召回率':<12}{'F1值':<12}")
print("-" * 90)

# 存储最终结果，方便你复制
final_result = {}

for method_name, sampler in sampling_method_list.items():
    auc_list = []
    recall_list = []
    f1_list = []

    # 5折交叉验证
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 采样处理
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # 训练模型
        model = base_model.fit(X_train, y_train)
        # 预测
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred_label = (y_pred_proba >= 0.5).astype(int)

        # 计算指标
        auc = roc_auc_score(y_val, y_pred_proba)
        recall = recall_score(y_val, y_pred_label)
        f1 = f1_score(y_val, y_pred_label)

        auc_list.append(auc)
        recall_list.append(recall)
        f1_list.append(f1)

    # 计算5折平均值
    mean_auc = np.mean(auc_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)

    # 保存结果
    final_result[method_name] = {
        "AUC": round(mean_auc, 4),
        "召回率": round(mean_recall, 4),
        "F1值": round(mean_f1, 4)
    }

    # 控制台输出，直接复制到论文
    print(f"{method_name:<30}{mean_auc:<12.4f}{mean_recall:<12.4f}{mean_f1:<12.4f}")

print("=" * 90)
print("✅ 实验完成！结果已输出，可直接复制到论文中")
print("=" * 90)