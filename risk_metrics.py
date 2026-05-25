import numpy as np
import pandas as pd
import matplotlib
# 关键修复：禁用图形界面，彻底解决报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, recall_score, f1_score, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")

# 数据加载
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"]).values
y = df["default payment next month"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2026,stratify=y)

# 模型训练
sampler = SMOTEENN(random_state=2026)
X_train, y_train = sampler.fit_resample(X_train, y_train)
model = LGBMClassifier(random_state=2026, verbose=-1)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:,1]
y_bin = (y_pred>=0.5).astype(int)

# 计算核心风控指标
ks = ks_2samp(y_pred[y_test==1], y_pred[y_test==0]).statistic
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = -np.trapz(precision, recall)
auc = roc_auc_score(y_test, y_pred)
recall_val = recall_score(y_test, y_bin)
f1_val = f1_score(y_test, y_bin)

# 保存混淆矩阵（无界面）
cm = confusion_matrix(y_test, y_bin)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.savefig("model_results/混淆矩阵.png",dpi=300,bbox_inches='tight')
plt.close()

# 保存PR曲线（无界面）
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f'PR-AUC={pr_auc:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("model_results/PR曲线.png",dpi=300)
plt.close()

# 输出真实结果
print("="*50)
print("风控核心指标（真实运行结果）")
print(f"AUC     : {auc:.4f}")
print(f"Recall  : {recall_val:.4f}")
print(f"F1      : {f1_val:.4f}")
print(f"KS      : {ks:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print("="*50)