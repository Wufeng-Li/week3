import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats

# # deal with crude data
# pag_with_dummy = df(pd.read_csv('pag_with_dummy.txt'))
# #pag_with_dummy1 = pag_with_dummy.drop(['device_id', 'label'], axis=1)
# data = pag_with_dummy
# print(data.head(3))
# print('')
# print("原始数据量： ", data.shape[0])
# print('')
#
# # delete abnormal data >> data_final
# data = data[data['daka'] < 1000]
# data = data[data['guide'] < 3000]
# data = data[data['index'] < 10000]
# data = data[data['note'] < 10000]
# data = data[data['other'] < 3000]
# data = data[data['poi'] < 3000]
# data = data[data['qa'] < 3000]
# #data = data[data['route'] < 31]
# data = data[data['search'] < 10000]
# #data = data[data['video'] < 31]
# data = data[data['weng'] < 10000]
# print("初删之后数据量： ", data.shape[0])
# print('')
#
# data_old = data.drop(['device_id', 'label'], axis=1)
# data_new = data_old.drop(data_old.columns[11:], axis=1)
# #print(data_new.head(5))
#
# # deal with IsolationForest
# ratio = 0.05
# clf = IsolationForest(
#     n_estimators = 100, # 基本估算器数量default=>100
#     max_samples = 100, # 最大样本数
#     contamination = ratio, # 数据集中异常值的比例
#     max_features = 1, # 从X中绘制以训练每个基本估计器的特征数
#     bootstrap = False, # 单个树适合随替换采样的训练数据的随机子集,即是否执行替换采样
#     n_jobs = -1,
#     behaviour = 'new',
#     random_state = 333,
# )
# clf.fit(data_new)
# scores_pred = clf.decision_function(data_new)
# threshold = stats.scoreatpercentile(scores_pred, 100 * ratio)
# print('')
# print("scores_pred type:", type(scores_pred))
# print('')
# print("scores_pred: ", scores_pred)
# print('')
# print("threshold: ", threshold)
# print('')
#
# scores_pred_df = pd.DataFrame(scores_pred, columns = ["scores_pred"])
# print("scores_pred_df type:", type(scores_pred_df))
# print('')
#
# # 根据训练样本中异常样本比例，得到阈值，用于绘图
# data_1 = pd.concat([data, scores_pred_df], axis=1, join_axes=[data.index])
# print(data_1.head(3))
# print('')
#
# data_2 = data_1[data_1['scores_pred'] > threshold]
# print("IsolationForest删除之后数据量： ", data_2.shape[0])
# print('')
# data_final = data_2.drop(["scores_pred"], axis=1)
# print(data_final.head(3))
# print('')
# data_final.to_csv('pag_with_dummy_if1.txt', index = False)
#
# print(data_final.head())
pag_with_dummy_if1 = df(pd.read_csv('pag_with_dummy_if1.txt'))
data_final = pag_with_dummy_if1.drop(["device_id"], axis=1)
#StandardScaler crude data
ss = StandardScaler()
data_final_=ss.fit_transform(data_final)
data_final_regular = pd.DataFrame(data_final_)

# find out k
SSE = []
for k in range(1, 10):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(data_final_regular)
    SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
    print("k = ", k, " SSE = ",estimator.inertia_)
# print(SSE)
# print(X.shape())
X = list(range(1, 10))
# X1 = [_ for _ in range(1, 10)]
# print("X:", X)
# print("X1:", X1)
# print(X.shape())
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.savefig("elbow_month3_last.png")
# plt.show()
