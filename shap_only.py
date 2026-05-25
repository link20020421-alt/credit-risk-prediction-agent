# encoding:utf8
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0.,inplace=True)

label = np.where(data_set['label'].values >= 0.5, 1, 0).astype(int)
feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')
training = data_set[feature_list].values

# 2. 训练极简模型
model = lgb.LGBMClassifier(
    objective='binary',
    max_depth=5,
    num_leaves=21,
    learning_rate=0.02,
    n_estimators=200,
    random_state=2017
)
model.fit(training[:5000], label[:5000])

# 3. SHAP 画图（强制弹出）
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(training[:2000])

shap.summary_plot(shap_values, training[:2000], feature_names=feature_list)
plt.savefig('SHAP_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP 图已生成！")