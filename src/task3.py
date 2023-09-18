#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
mp.matplotlib_fname()

from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
import sklearn.metrics as sm

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.notebook_repr_html', False)


# In[2]:


excel1 = pd.read_excel(io='../data/附件1.xlsx', sheet_name='Sheet1')

def df2List(tabel):
    df = excel1[[tabel]]
    _array = np.array(df.stack())
    _list = _array.tolist()
    return _list

# 单品名称列表
name_list = df2List('单品名称')
# 分类名称列表
class_name_list = df2List('分类名称')

# 单品名称2分类名称哈希表
name2class = {}
for i in range(len(name_list)):
    name2class[name_list[i]] = class_name_list[i]


# In[3]:


weight = pd.read_csv('../data/sell_weight.csv')[-283::]
earn = pd.read_csv('../data/earn_daily.csv')[-283::]
# 创建单品编码与单品名称哈希表
lost = pd.read_excel(io='../data/附件4.xlsx', sheet_name='Sheet1')


# In[4]:


need = pd.DataFrame({
    "水生根茎类":[10.618191],   
    "花叶类": [185.167241],
    "花菜类": [41.569375],
    "茄类": [28.143449], 
    "辣椒类": [56.647898],
    "食用菌": [42.880418]
})
need


# In[5]:


weight_total = weight.drop('销售日期', axis=1).groupby('单品编码', as_index=False).sum()
earn_total = earn.drop('销售日期', axis=1).groupby('单品编码', as_index=False).sum()
weight_total, earn_total


# In[6]:


earn_rate_df = pd.DataFrame()
earn_rate_df['单品编码'] = weight_total['单品编码']
earn_rate_df['利润率(元/千克)'] = earn_total['利润(元)'] / weight_total['销量(千克)']


# In[7]:


earn_rate_df = earn_rate_df.sort_values(by="利润率(元/千克)", ascending=False)
earn_rate_df = earn_rate_df.reset_index(drop=True)
earn_rate_df


# In[8]:


need_list = [["水生根茎类", "花叶类", "花菜类", "茄类", "辣椒类", "食用菌"],
             [10.618191, 185.167241, 41.569375, 28.143449, 56.647898, 42.880418]]


# In[9]:


buy = pd.DataFrame(columns=['单品名称', '进货量(千克)'])
buy_list = [0, 0, 0, 0, 0, 0]
j = 0
while buy_list != need_list[1]:
    i = j % 49
    veg_index = need_list[0].index(name2class[earn_rate_df.loc[i]['单品编码']])
    if buy_list[veg_index] != need_list[1][veg_index]:
        if need_list[1][veg_index] - buy_list[veg_index] >= 2.5:
            buy_list[veg_index] += 2.5
            buy.loc[len(buy.index)] = [earn_rate_df.loc[i]['单品编码'], 2.5]
        elif need_list[1][veg_index] - buy_list[veg_index] == 0:
            j+=1
            continue
        elif need_list[1][veg_index] - buy_list[veg_index] < 2.5:
            buy.loc[len(buy.index)] = [earn_rate_df.loc[i]['单品编码'], need_list[1][veg_index] - buy_list[veg_index]]
            buy_list[veg_index] += need_list[1][veg_index] - buy_list[veg_index]
    j+=1

buy = buy.groupby('单品名称', as_index=False).sum()
buy.style


# 综上，洪湖藕带不足2.5千克，舍去，换位同品类利润率最大的单品

# In[10]:


for i in range(len(earn_rate_df)):
    if name2class[earn_rate_df.loc[i]['单品编码']] == name2class['洪湖藕带']:
        buy.loc[len(buy.index)] = [earn_rate_df.loc[i]['单品编码'], buy.loc[19].values.tolist()[1]]
        break
buy = buy.drop(19).groupby('单品名称', as_index=False).sum()
buy = pd.merge(buy,lost,on='单品名称',how='inner').drop('单品编码', axis=1)
buy.style


# In[45]:

earn_total.sort_values(by="利润(元)", ascending=False).set_index('单品编码').plot.bar()
plt.show()

earn_rate_df.set_index('单品编码').plot.bar()
plt.show()

