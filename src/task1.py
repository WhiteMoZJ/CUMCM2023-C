#!/usr/bin/env python
# coding: utf-8

# # 导入数据 预处理

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
mp.matplotlib_fname()
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:


pd.set_option('display.notebook_repr_html', False)
data = pd.read_excel(io='../data/附件2.xlsx', sheet_name='Sheet1')


# In[3]:


data


# ## 创建单品编码与单品名称 单品名称与其分类哈希表

# In[4]:


excel1 = pd.read_excel(io='../data/附件1.xlsx', sheet_name='Sheet1')
excel1


# In[5]:


def df2List(tabel):
    df = excel1[[tabel]]
    _array = np.array(df.stack())
    _list = _array.tolist()
    return _list


# In[6]:


# 单品编码列表
code_list = df2List('单品编码')
# 单品名称列表
name_list = df2List('单品名称')
# 分类编码列表
class_code_list = df2List('分类编码')
# 分类名称列表
class_name_list = df2List('分类名称')


# In[7]:


# 单品编码2单品名称哈希表
code2name = {}
for i in range(len(code_list)):
    code2name[code_list[i]] = name_list[i]
code2name


# In[8]:


# 单品名称2分类名称哈希表
name2class = {}
for i in range(len(name_list)):
    name2class[name_list[i]] = class_name_list[i]
name2class


# In[9]:


sell_amount = data[['销售日期', '单品编码', '销量(千克)']]
sell_amount.loc[:,'销售日期'] = pd.to_datetime(sell_amount.loc[:,'销售日期'])
for i in range(len(sell_amount)):
    sell_amount.loc[i,'单品编码'] = code2name[sell_amount.loc[i,'单品编码']]
sell_amount


# 不同单品总销量

# In[10]:


sell_amount_total = sell_amount.groupby('单品编码')['销量(千克)'].sum()
sell_amount_total


# In[11]:


# sell_amount_total.plot.bar(color='blue', title='不同单品蔬菜销售总量')
# plt.xlabel('单品')
# plt.ylabel('销售量（千克）')
# plt.show()


# In[12]:


sell_monthly = sell_amount.groupby(['单品编码', sell_amount['销售日期'].dt.year, sell_amount['销售日期'].dt.month])['销量(千克)'].sum()
monthly_sell = sell_amount.groupby([sell_amount['销售日期'].dt.year, sell_amount['销售日期'].dt.month, '单品编码'])['销量(千克)'].sum()


# 不同单品月销售量

# In[13]:


sell_monthly


# ## 每月不同单品销售量

# In[14]:


sell_class_amount = sell_amount.copy(deep=True)
for i in range(len(sell_class_amount)):
    sell_class_amount.loc[i,'单品编码'] = name2class[sell_amount.loc[i,'单品编码']]
sell_class_amount


# ## 不同品类总销量

# In[15]:


sell_class_total = sell_class_amount.groupby('单品编码')['销量(千克)'].sum()
sell_class_total


# In[16]:


sell_class_total.plot.pie(autopct='%.2f%%')
plt.xlabel('品类')
plt.ylabel('销售量（千克）')
plt.show()


# In[17]:


sell_class_total.plot.bar()
plt.xlabel('品类')
plt.ylabel('销售量（千克）')
plt.show()


# ## 不同品类月销量

# In[18]:


sell_class_amount_monthly = sell_class_amount.groupby(['单品编码', sell_amount['销售日期'].dt.year, sell_amount['销售日期'].dt.month])['销量(千克)'].sum()
sell_monthly_class_amount = sell_class_amount.groupby([sell_amount['销售日期'].dt.year, sell_amount['销售日期'].dt.month, '单品编码'])['销量(千克)'].sum()


# In[19]:


sell_class_amount_monthly


# In[20]:


# 品类月销售图表
for i in sell_class_total.index:
    sell_class_amount_monthly[i].plot(label=i)
plt.title("品类月销售图表")
plt.xlabel('销售年/月')
plt.ylabel('销售量（千克）')
plt.legend(loc='best')
plt.show()


# # 相关性分析

# ## 品类

# In[21]:


aq_rhi = [] # 水生根茎类
flo_leaves = [] # 花叶类
flo_veg = [] # 花菜类
solanula = [] # 茄类
peppers = [] # 辣椒类
fungi = [] # 食用菌

j = 1
for i in sell_class_amount_monthly:
    if j <= 36:
        aq_rhi.append(i)
    elif j <= 72:
        flo_leaves.append(i)
    elif j <= 108:
        flo_veg.append(i)
    elif j <= 144:
        solanula.append(i)
    elif j <= 180:
        peppers.append(i)
    else:
        fungi.append(i)
    j+=1


# 归一化处理

# In[22]:


scaler = StandardScaler()
aq_rhi_n = scaler.fit_transform(np.array(aq_rhi).reshape(-1, 1)).tolist()
flo_leaves_n = scaler.fit_transform(np.array(flo_leaves).reshape(-1, 1)).tolist()
flo_veg_n = scaler.fit_transform(np.array(flo_veg).reshape(-1, 1)).tolist()
solanula_n = scaler.fit_transform(np.array(solanula).reshape(-1, 1)).tolist()
peppers_n = scaler.fit_transform(np.array(peppers).reshape(-1, 1)).tolist()
fungi_n = scaler.fit_transform(np.array(fungi).reshape(-1, 1)).tolist()


# In[23]:


sell = [aq_rhi_n, flo_leaves_n, flo_veg_n, solanula_n, peppers_n, fungi_n]
sell_name = ["水生根茎类", "花叶类", "花菜类", "茄类", "辣椒类", "食用菌"]


# In[24]:


relate_class = np.zeros([6,6])


# 使用Spearmanr秩相关系数

# In[25]:


for i in range(6):
    for j in range(6):
        relate_class[i, j], p = spearmanr(sell[i], sell[j])
relate_class


# In[26]:


plt.xticks(np.arange(len(sell)), labels=sell_name, 
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell)), labels=sell_name)
plt.title("不同品类蔬菜销售量相关系数")
for i in range(len(sell)):
    for j in range(len(sell)):
        text = plt.text(j, i, '%.2f'%relate_class[i, j], ha="center", va="center")
plt.imshow(relate_class, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# ## 同品类间的相关性(按月分布)

# In[27]:


sell_amount_total = sell_amount.groupby('单品编码')['销量(千克)'].sum().to_frame()
sell_amount_total_sort = sell_amount_total.sort_values(by="销量(千克)",ascending=False)
sell_amount_total_top = sell_amount_total_sort.index.tolist()[0:int(len(sell_amount_total_sort.index) * 0.1)]
sell_monthly_df = sell_monthly.unstack(0).fillna(0)
sell_monthly_df


# In[28]:


sell_leave_total_top_list = []
sell_leave_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == "花叶类":
        sell_leave_total_top_list.append(sell_monthly_df[i].values)
        sell_leave_total_top_name.append(i)

relate_leave = np.zeros([len(sell_leave_total_top_name), len(sell_leave_total_top_name)])
for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_leave_total_top_list[i], sell_leave_total_top_list[j])

plt.figure(figsize=(15, 15))
plt.xticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name, 
                     rotation=90, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name)
plt.title("不同花叶类蔬菜销售量相关系数")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# In[29]:


sell_leave_total_top_list = []
sell_leave_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == "水生根茎类":
        sell_leave_total_top_list.append(sell_monthly_df[i].values)
        sell_leave_total_top_name.append(i)

relate_leave = np.zeros([len(sell_leave_total_top_name), len(sell_leave_total_top_name)])
for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_leave_total_top_list[i], sell_leave_total_top_list[j])

plt.figure(figsize=(10, 10))
plt.xticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name, 
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name)
plt.title("不同水生根茎类蔬菜销售量相关系数")

for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        text = plt.text(j, i, '%.2f'%relate_leave[i, j], ha="center", va="center")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# In[30]:


sell_leave_total_top_list = []
sell_leave_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == "花菜类":
        sell_leave_total_top_list.append(sell_monthly_df[i].values)
        sell_leave_total_top_name.append(i)

relate_leave = np.zeros([len(sell_leave_total_top_name), len(sell_leave_total_top_name)])
for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_leave_total_top_list[i], sell_leave_total_top_list[j])

plt.xticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name, 
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name)
plt.title("不同花菜类蔬菜销售量相关系数")

for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        text = plt.text(j, i, '%.2f'%relate_leave[i, j], ha="center", va="center")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# In[31]:


sell_leave_total_top_list = []
sell_leave_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == "茄类":
        sell_leave_total_top_list.append(sell_monthly_df[i].values)
        sell_leave_total_top_name.append(i)

relate_leave = np.zeros([len(sell_leave_total_top_name), len(sell_leave_total_top_name)])
for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_leave_total_top_list[i], sell_leave_total_top_list[j])

plt.figure(figsize=(6, 6))
plt.xticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name, 
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name)
plt.title("不同茄类蔬菜销售量相关系数")

for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        text = plt.text(j, i, '%.2f'%relate_leave[i, j], ha="center", va="center")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# In[32]:


sell_leave_total_top_list = []
sell_leave_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == "辣椒类":
        sell_leave_total_top_list.append(sell_monthly_df[i].values)
        sell_leave_total_top_name.append(i)

relate_leave = np.zeros([len(sell_leave_total_top_name), len(sell_leave_total_top_name)])
for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_leave_total_top_list[i], sell_leave_total_top_list[j])

plt.figure(figsize=(15, 15))
plt.xticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name, 
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name)
plt.title("不同辣椒类蔬菜销售量相关系数")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# In[33]:


sell_leave_total_top_list = []
sell_leave_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == "食用菌":
        sell_leave_total_top_list.append(sell_monthly_df[i].values)
        sell_leave_total_top_name.append(i)

relate_leave = np.zeros([len(sell_leave_total_top_name), len(sell_leave_total_top_name)])
for i in range(len(sell_leave_total_top_name)):
    for j in range(len(sell_leave_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_leave_total_top_list[i], sell_leave_total_top_list[j])

plt.figure(figsize=(15, 15))
plt.xticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name, 
                     rotation=90, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_leave_total_top_name)), labels=sell_leave_total_top_name)
plt.title("不同食用菌蔬菜销售量相关系数")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()


# # 不同品类相关性

# In[34]:


sell_total_top_list = []
sell_total_top_name = []
for i in sell_monthly_df:
    if name2class[i] == '水生根茎类' and len(sell_total_top_name) <=10:
        sell_total_top_list.append(sell_monthly_df[i].values)
        sell_total_top_name.append(i)
        continue
    if name2class[i] == '花叶类' and len(sell_total_top_name) <=20:
        sell_total_top_list.append(sell_monthly_df[i].values)
        sell_total_top_name.append(i)
        continue
    if name2class[i] == '花菜类' and len(sell_total_top_name) <=25:
        sell_total_top_list.append(sell_monthly_df[i].values)
        sell_total_top_name.append(i)
        continue
    if name2class[i] == '茄类' and len(sell_total_top_name) <=35:
        sell_total_top_list.append(sell_monthly_df[i].values)
        sell_total_top_name.append(i)
        continue
    if name2class[i] == '辣椒类' and len(sell_total_top_name) <=45:
        sell_total_top_list.append(sell_monthly_df[i].values)
        sell_total_top_name.append(i)
        continue
    if name2class[i] == '食用菌' and len(sell_total_top_name) <=55:
        sell_total_top_list.append(sell_monthly_df[i].values)
        sell_total_top_name.append(i)

relate_leave = np.zeros([len(sell_total_top_name), len(sell_total_top_name)])
for i in range(len(sell_total_top_name)):
    for j in range(len(sell_total_top_name)):
        relate_leave[i, j], p = spearmanr(sell_total_top_list[i], sell_total_top_list[j])

plt.figure(figsize=(15, 15))
plt.xticks(np.arange(len(sell_total_top_name)), labels=sell_total_top_name, 
                     rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(len(sell_total_top_name)), labels=sell_total_top_name)
plt.title("不同品类蔬菜销售量相关系数")

plt.imshow(relate_leave, cmap='coolwarm', origin='upper')
plt.tight_layout()
plt.colorbar()
plt.show()

