import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# deal with crude data
pag_with_dummy = df(pd.read_csv('pag_with_dummy.txt'))
pag_with_dummy1 = pag_with_dummy.drop(['device_id', 'label'], axis=1)

# StandardScaler crude data
ss = StandardScaler()
pag_with_dummy1_regular=ss.fit_transform(pag_with_dummy1)
 
# find out k
SSE = []
for k in range(1, 10):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(pag_with_dummy1_regular)
    SSE.append(estimator.inertia_)# estimator.inertia_获取聚类准则的总和
    rint("k = ", k, " SSE = ",estimator.inertia_)
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
plt.savefig("elbow.png")
# plt.show()



