import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cdist

# deal with crude data
pag_with_dummy = df(pd.read_csv('pag_with_dummy.txt'))
data = pag_with_dummy.drop(['device_id', 'label'], axis=1)

# delete abnormal data >> data_final
data = data[data['daka'] < 1000]
data = data[data['guide'] < 3000]
data = data[data['index'] < 10000]
data = data[data['note'] < 10000]
data = data[data['other'] < 3000]
data = data[data['poi'] < 3000]
data = data[data['qa'] < 3000]
#data = data[data['route'] < 31]
data = data[data['search'] < 10000]
#data = data[data['video'] < 31]
data = data[data['weng'] < 10000]

data_final = data

# based on previous analysis, the best k=4
# the process of KMeans as follow
num_clusters = 3
estimator = KMeans(n_clusters=num_clusters, # 构造聚类器
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=0.0001,
            precompute_distances='auto',
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=-1,
            algorithm='auto'
            )
estimator.fit(data_final) # 聚类
label_pred = estimator.labels_  #  获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

# print result
r1 = pd.Series(estimator.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(estimator.cluster_centers_) #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（axis=0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data_final.columns) + [u'类别数目'] #重命名表头
print(" ")
print("k=3 and delete abnormal data ")
print(r)
print(" ")
print("inertia: ", inertia)
