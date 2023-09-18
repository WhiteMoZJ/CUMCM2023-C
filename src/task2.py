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


# In[2]:


pd.set_option('display.notebook_repr_html', False)
data = pd.read_excel(io='../data/附件2.xlsx', sheet_name='Sheet1')


# In[3]:


# 创建单品编码与单品名称哈希表
excel1 = pd.read_excel(io='../data/附件1.xlsx', sheet_name='Sheet1')

def df2List(tabel):
    df = excel1[[tabel]]
    _array = np.array(df.stack())
    _list = _array.tolist()
    return _list

# 单品编码列表
code_list = df2List('单品编码')
# 单品名称列表
name_list = df2List('单品名称')
# 分类编码列表
class_code_list = df2List('分类编码')
# 分类名称列表
class_name_list = df2List('分类名称')

# 单品编码2单品名称哈希表
code2name = {}
for i in range(len(code_list)):
    code2name[code_list[i]] = name_list[i]
# 单品名称2分类名称哈希表
name2class = {}
for i in range(len(name_list)):
    name2class[name_list[i]] = class_name_list[i]


# In[4]:


lost = pd.read_excel(io='../data/附件4.xlsx', sheet_name='Sheet1')
lost_mean = lost.copy(deep=True)
for i in range(len(lost_mean)):
    lost_mean.loc[i,'单品名称'] = name2class[lost.loc[i,'单品名称']]

lost_mean = lost_mean.groupby('单品名称', as_index=False)['损耗率(%)'].mean()
lost_mean.style


# In[5]:


sell_daily = data[['销售日期', '单品编码', '销量(千克)', '销售单价(元/千克)']]
sell_daily.loc[:,'销售日期'] = pd.to_datetime(sell_daily.loc[:,'销售日期'])

for i in range(len(sell_daily)):
    sell_daily.loc[i,'单品编码'] = code2name[sell_daily.loc[i,'单品编码']]
sell_daily


# In[6]:


cost = pd.read_excel(io='../data/附件3.xlsx', sheet_name='Sheet1')
for i in range(len(cost)):
    cost.loc[i,'单品编码'] = code2name[cost.loc[i,'单品编码']]
cost.rename(columns={'日期': '销售日期'}, inplace=True)
cost


# In[7]:


price_cost_df = pd.merge(sell_daily, cost, on=['销售日期','单品编码'],how='inner')
price_cost_df = price_cost_df.groupby(['销售日期', '单品编码', '销售单价(元/千克)', '批发价格(元/千克)'], as_index=False).sum()
price_cost_df


# In[8]:


earn_daily = pd.DataFrame()
earn_daily['销售日期'] = price_cost_df['销售日期']
earn_daily['单品编码'] = price_cost_df['单品编码']
earn_daily['利润(元)'] = price_cost_df['销量(千克)'] * (price_cost_df['销售单价(元/千克)'] - price_cost_df['批发价格(元/千克)'])
earn_daily = earn_daily.groupby(['销售日期', '单品编码'], as_index=False).sum()
earn_daily


# ### 每日利润

# In[9]:


class_earn_daily = earn_daily.copy(deep=True)
for i in range(len(class_earn_daily)):
    class_earn_daily.loc[i,'单品编码'] = name2class[class_earn_daily.loc[i,'单品编码']]
class_earn_daily = class_earn_daily.groupby(['单品编码', class_earn_daily['销售日期']])['利润(元)'].sum()
class_earn_daily = class_earn_daily.unstack(0).fillna(0)
class_earn_daily


# ### 每日销售额

# In[10]:


value_daily = pd.DataFrame()
value_daily['销售日期'] = price_cost_df['销售日期']
value_daily['单品编码'] = price_cost_df['单品编码']
value_daily['销售额(元)'] = price_cost_df['销量(千克)'] * price_cost_df['销售单价(元/千克)']
value_daily = value_daily.groupby(['销售日期', '单品编码'], as_index=False).sum()
value_daily


# In[11]:


class_value_daily = value_daily.copy(deep=True)
for i in range(len(class_value_daily)):
    class_value_daily.loc[i,'单品编码'] = name2class[class_value_daily.loc[i,'单品编码']]
class_value_daily = class_value_daily.groupby(['单品编码', class_value_daily['销售日期']])['销售额(元)'].sum()
class_value_daily = class_value_daily.unstack(0).fillna(0)
class_value_daily


# ### 品类批发成本

# In[12]:


cost_daily = pd.DataFrame()
cost_daily['销售日期'] = price_cost_df['销售日期']
cost_daily['单品编码'] = price_cost_df['单品编码']
cost_daily['批发价(元)'] = price_cost_df['销量(千克)'] * price_cost_df['批发价格(元/千克)']
cost_daily = cost_daily.groupby(['销售日期', '单品编码'], as_index=False).sum()
cost_daily


# In[13]:


class_cost_daily = cost_daily.copy(deep=True)
for i in range(len(class_cost_daily)):
    class_cost_daily.loc[i,'单品编码'] = name2class[class_cost_daily.loc[i,'单品编码']]
class_cost_daily = class_cost_daily.groupby(['单品编码', class_cost_daily['销售日期']])['批发价(元)'].sum()
class_cost_daily = class_cost_daily.unstack(0).fillna(0)
class_cost_daily


# ### 每日销售量

# In[14]:


class_sell_daily = sell_daily.copy(deep=True)
class_sell_daily.loc[:,'销售日期'] = pd.to_datetime(class_sell_daily.loc[:,'销售日期'])
class_sell_daily.drop('销售单价(元/千克)', axis=1)
for i in range(len(class_sell_daily)):
    class_sell_daily.loc[i,'单品编码'] = name2class[class_sell_daily.loc[i,'单品编码']]
class_sell_daily = class_sell_daily.groupby(['单品编码', class_sell_daily['销售日期']])['销量(千克)'].sum()
class_sell_daily = class_sell_daily.unstack(0).fillna(0)
class_sell_daily


# ### 每日批发价

# In[15]:


class_cost_in_daily = class_cost_daily / class_sell_daily
class_cost_in_daily


# ### 每日平均售价

# In[16]:


class_value_in_daily = class_value_daily / class_sell_daily
class_value_in_daily


# ### 成本加成比例

# In[17]:


cost_addition = pd.DataFrame()
cost_addition = class_value_daily/(class_value_daily - class_earn_daily) - 1


# In[18]:


cost_addition


# In[19]:


class_cost_daily.plot(label=class_sell_daily.index)
plt.title("品类月批发价图表")
plt.xlabel('销售年/月')
plt.ylabel('成本(元)')
class_cost_in_daily.plot(label=class_sell_daily.index)
plt.title("品类月批发单价图表")
plt.xlabel('销售年/月')
plt.ylabel('批发价(元/千克)')
class_value_daily.plot(label=class_earn_daily.index)
plt.title("品类月销售额图表")
plt.xlabel('销售年/月')
plt.ylabel('销售额(元)')
class_value_in_daily.plot(label=class_sell_daily.index)
plt.title("品类月销售单价图表")
plt.xlabel('销售年/月')
plt.ylabel('售价(元/千克)')
class_sell_daily.plot(label=class_sell_daily.index)
plt.title("品类月销售量图表")
plt.xlabel('销售年/月')
plt.ylabel('销售量(千克)')
class_earn_daily.plot(label=class_earn_daily.index)
plt.title("品类月利润图表")
plt.xlabel('销售年/月')
plt.ylabel('利润(元)')
plt.legend(loc='best')
plt.show()
cost_addition.plot(label=cost_addition.index)
plt.title("成本加成比例")
plt.xlabel('销售年/月')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()


# In[20]:


cost_addition[-30::].plot(label=cost_addition.index)
plt.title("近30天成本加成比例")
plt.xlabel('销售年/月')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()


# 代价函数
# $$
# J(w,b)=\frac{1}{m}\sum^{m}_{i=1}|f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}|
# $$

# In[21]:


def CostFunc(y, pred_y):
    J = 0
    for i in range(len(y)):
        J += abs(pred_y[i] - y[i])
    J = J / len(y)
    return J


# ### 基于ARIMA模型

# In[22]:


cost_pre = np.array([])

j = 231
relate = []
cost = []

plt.figure(figsize=(15, 8))
for i in range(len(cost_addition.columns.tolist())):
    name = cost_addition.columns.tolist()[i]
    plt.subplot(j)
    plt.title(name)
    plt.ylabel('平均批发价')

    # 原始值与滤波值
    x = [*range(len(class_cost_in_daily[name]))]
    data = class_cost_in_daily[name].to_frame().fillna(class_cost_in_daily[name].mean())
   
    ts_diff = data.diff(1)
    ts_diff.dropna(inplace=True)

    ts_diff = ts_diff.diff(1)
    ts_diff.dropna(inplace=True)

    # 拟合ARIMA模型
    model = ARIMA(data, order=(1, 1, 1))
    results_ARIMA = model.fit()
    
    # 绘制拟合数据
    plt.scatter(x, data.values.reshape(1, -1).tolist()[0], s=1, label='原始值')
    plt.plot(results_ARIMA.fittedvalues.values, color='orange', label='基于模型拟合原始数据')
    plt.plot(results_ARIMA.predict(1085, 1092, dynamic=True), color='red', label='预测值')
    j+=1

    pred_y = results_ARIMA.fittedvalues.fillna(0).to_frame().values.reshape(1, -1).tolist()[0]
    data_list = data.values.reshape(1, -1).tolist()[0]

    v, p = spearmanr(data_list, pred_y)
    cost.append(CostFunc(pred_y, data_list))
    relate.append(v)
    cost_pre = np.append(cost_pre, results_ARIMA.predict(1086, 1092, dynamic=True).to_frame().values.tolist())
    
    
plt.legend(loc='best')    
plt.show()

plt.figure(figsize=(6, 4))
plt.ylim(0,1)
plt.bar(cost_addition.columns.tolist(), relate, lw=0.5,fc="blue",width=0.3)

plt.title("近一个月品类成本加成比例原始值与预测值相关性图表")
plt.xlabel('品类')
plt.ylabel('相关系数')
plt.show()

print(relate)

plt.figure(figsize=(6, 4))
plt.bar(cost_addition.columns.tolist(), cost, lw=0.5,fc="r",width=0.3)

plt.title("平均绝对误差MAE")
plt.xlabel('品类')
plt.ylabel('MAE')
plt.show()

print(cost)


# In[23]:


cost_pre_df = pd.DataFrame(cost_pre.reshape(6, -1).T, columns = cost_addition.columns.to_list(), index=[*range(1, 8)])
cost_pre_df.style


# In[129]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# In[270]:


model = lm.LinearRegression()

cost = []
model_list = []

j = 321
plt.figure(figsize=(18, 15))
for i in range(len(cost_addition.columns.tolist())):
    name = cost_addition.columns.tolist()[i]
    ax1 = plt.subplot(j)
    plt.title(name)
    x = class_value_in_daily[name].fillna(class_value_in_daily[name].mean()).values
    y = class_sell_daily[name].fillna(class_sell_daily[name].mean()).values
    
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    pred_y = model.predict(x.reshape(-1, 1))
    ax1.set_ylabel('销售量(千克)')
    plt.scatter(x, y, s=1)
    plt.plot(x, pred_y, 'ro', label='Price Regression', markersize=1)
    plt.legend(loc=2)
    
    ax2 = ax1.twinx()
    plt.plot(x, pred_y.reshape(-1) * x.reshape(-1), 'yo', label='Sell Base on Price Regression', markersize=1)
    ax2.set_ylabel('销售额(元)') 
    plt.legend(loc=3)
    
    print('coef_:%.3f intercept_:%.3f' % (model.coef_, model.intercept_))
    model_list.append([model.coef_, model.intercept_])

    cost.append(CostFunc(pred_y, y).tolist()[0])
    j+=1

plt.legend(loc='best')
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(cost_addition.columns.tolist(), cost, lw=0.5,fc="r",width=0.3)

plt.title("平均绝对误差MAE")
plt.xlabel('品类')
plt.ylabel('MAE')
plt.show()


# 存在
# 当处于某种售价时的销售额关系式
# $$
# \begin{gather}
# S=aP+b\\
# P=C(1+\alpha)\\
# \end{gather}
# $$
# 且$a<0$
# 求利润$E$最大值
# $$
# \max(E)\iff\max(\alpha{S}=\alpha(a{C(1+\alpha)}+b))
# $$

# In[272]:


from scipy.optimize import fmin

def f(x, a, b, C):
    return x*(a*(C*(1+x))+b)

def fmax(func, x0, args=(), **kwargs):
    return fmin(lambda x: -func(x, *args), x0, **kwargs, disp=False)


add_list = []
for i in range(len(cost_addition.columns.to_list())):
    add_list.append([])
    for C in cost_pre_df[cost_addition.columns.to_list()[i]].tolist():
        # print('coef_:%.3f intercept_:%.3f cost:%.3f' % (model_list[i][0], model_list[i][1], C))
        result = fmax(f, 0.5, args=(model_list[i][0], model_list[i][1], C))
        add_list[i].append(result[0])

