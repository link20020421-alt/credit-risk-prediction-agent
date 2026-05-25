import pickle
import numpy as np

# 你的数据集路径，和之前train.py里的路径一致
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file, 'rb'))

label = np.where(data_set['label'].values >= 0.5, 1, 0).astype(int)

# 计算各类样本数量
total_num = len(label)
negative_num = len(label[label == 0])  # 正常样本
positive_num = len(label[label == 1])  # 违约样本

# 计算占比
negative_rate = negative_num / total_num * 100
positive_rate = positive_num / total_num * 100

# 输出结果，直接填进表格
print("="*30)
print(f"总样本数：{total_num}")
print(f"正常样本数：{negative_num}，占比：{negative_rate:.2f}%")
print(f"违约样本数：{positive_num}，占比：{positive_rate:.2f}%")
print("="*30)