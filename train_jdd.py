# encoding:utf8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# ====================== 数据加载 ======================
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0.,inplace=True)

label = np.where(data_set['label'].values >= 0.5, 1, 0).astype(int)
feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')
X = data_set[feature_list].values
y = label

test_data = pickle.load(open('./data/test.pkl','rb'))
test_data.fillna(0.,inplace=True)
sub_df = test_data['uid'].copy()
del test_data['uid']
test_data = test_data.values

# ====================== 5 折 + SMOTEENN ======================
skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)
auc_list, recall_list, f1_list = [], [], []
test_preds = []

# 用来保存最后一折的 X_val, y_val 用于画图
X_val_final = None
y_val_final = None
model_final = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

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

    test_preds.append(model.predict(test_data, num_iteration=model.best_iteration))

    # 保存最后一折数据画图用
    X_val_final = X_val
    y_val_final = y_val
    model_final = model

# ====================== 论文结果输出 ======================
print("\n" + "="*50)
print("           论文最终 5 折结果")
print("="*50)
print(f"AUC    : {np.mean(auc_list):.5f}")
print(f"召回率  : {np.mean(recall_list):.5f}")
print(f"F1     : {np.mean(f1_list):.5f}")
print("="*50)

# ====================== 提交文件 ======================
pred = np.mean(test_preds, axis=0)
pd.DataFrame({'uid': sub_df, 'pred': pred}).to_csv('submission.csv', index=False, header=None)
print("✅ 已生成 submission.csv")

# ====================== 【最终修复版】特征重要性图 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8), dpi=300)

ax = lgb.plot_importance(model_final, max_num_features=10)
ax.set_yticklabels(feature_list[:10])
plt.title("模型特征重要性排序")
plt.xlabel("特征重要性分数")
plt.ylabel("特征名称")

plt.savefig('特征重要性_业务名.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 特征重要性图已保存！")

# ====================== 【最终修复版】SHAP 蜂群图 ======================
explainer = shap.TreeExplainer(model_final)
shap_values = explainer.shap_values(X_val_final[:2000])

plt.figure(figsize=(12, 8), dpi=300)
shap.summary_plot(shap_values, X_val_final[:2000], feature_names=feature_list, show=False)
plt.savefig('SHAP_业务名_最终版.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP蜂群图已保存！")

# ====================== ✅ 【论文必须：图3 + 图4 SHAP力导向图】 ======================
X_test_df = pd.DataFrame(X_val_final, columns=feature_list)
y_test = y_val_final

# 预测概率
y_pred_proba = model_final.predict(X_val_final)

# 选样本
idx_default = np.where((y_test == 1) & (y_pred_proba > 0.9))[0][0]
idx_normal = np.where((y_test == 0) & (y_pred_proba < 0.1))[0][0]

# 画图
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 图3
plt.figure(figsize=(18, 4), dpi=300)
shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(X_val_final)[idx_default],
    X_test_df.iloc[idx_default],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("图3_违约样本SHAP力导向图.png", bbox_inches='tight', dpi=300)
plt.close()

# 图4
plt.figure(figsize=(18, 4), dpi=300)
shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(X_val_final)[idx_normal],
    X_test_df.iloc[idx_normal],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("图4_正常样本SHAP力导向图.png", bbox_inches='tight', dpi=300)
plt.close()

print("\n✅ ✅ ✅ 全部成功生成！图3、图4已保存！")