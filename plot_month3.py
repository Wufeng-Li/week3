import pandas as pd
import matplotlib as plt
import numpy as np
import string
import os
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from pandas import Series
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', 100)

path = '/home/liwufeng/week3/data'
data_label = df(pd.read_csv(path + 'data_label4.txt'))

label_0 = data_label[data_label['pred'] == 0]
label_1 = data_label[data_label['pred'] == 1]
label_2 = data_label[data_label['pred'] == 2]
label_3 = data_label[data_label['pred'] == 3]
# print(label_2.head())
for k in range(1,13):
    col_name = data_label.columns[k]
    plt.figure(figsize=(4,2),dpi=100)
    sns.distplot(data_label[col_name], hist=False, kde=True, rug=True,
                 kde_kws={"color":"b", "lw":1.5, 'linestyle':'--'},
                 rug_kws={'color':'b','alpha':1, 'lw':2,}, label=col_name+'_total')
    sns.distplot(label_0[col_name], hist=False, kde=True, rug=True,
                 kde_kws={"color":"r", "lw":1.5, 'linestyle':'--'},
                 rug_kws={'color':'r','alpha':1, 'lw':2,}, label=col_name+'_0')
    sns.distplot(label_1[col_name], hist=False, kde=True, rug=True,
                 kde_kws={"color":"g", "lw":1.5, 'linestyle':'--'},
                 rug_kws={'color':'g','alpha':1, 'lw':2,}, label=col_name+'_1')
    sns.distplot(label_2[col_name], hist=False, kde=True, rug=True,
                 kde_kws={"color":"m", "lw":1.5, 'linestyle':'--'},
                 rug_kws={'color':'m','alpha':1, 'lw':2,}, label=col_name+'_2')
    sns.distplot(label_3[col_name], hist=False, kde=True, rug=True,
                 kde_kws={"color":"lightcoral", "lw":1.5, 'linestyle':'--'},
                 rug_kws={'color':'lightcoral','alpha':1, 'lw':2,}, label=col_name+'_3')
    plt.savefig(col_name)