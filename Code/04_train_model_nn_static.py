# -*- coding: utf-8 -*-
"""
【华泰金工】日频多因子系统

模型训练

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd

import data.dataset as dataset

import model.pytorch_nn as pytorch_nn

import model.utils as utils


#%% 参数 
d_feat = 10

path = {
    'factors': '../Data/factors/', # 因子存放路径
    'results':'../Results/nn/', # 结果存放路径
}
os.makedirs(path['results'],exist_ok=True)


#%% 创建Dataset
# 读取csv
raw_data = pd.read_csv(path['factors']+'df_processed.csv',index_col=[0,1])

# 创建Dataset
ds = dataset.DatasetH(raw_data,
                      factors=raw_data.columns[:-3].tolist(),
                      labels=[raw_data.columns[-1]],
                      style_factors=raw_data.columns[-3:-1].tolist(),
                      leaky=10+1)

del raw_data


#%% 切分训练、验证、测试集
segments = {'train':('2011-01-01','2015-12-31'),
            'valid':('2016-01-01','2016-12-31'),
            'test':('2017-01-01','2022-09-30')}
ds.split(segments)


#%% 设置模型
model = pytorch_nn.NN(
    d_feat=d_feat, # 因子数
    hidden_size=64, # 隐状态数 64
    n_epochs=200, # 迭代次数
    lr=0.0001, # 学习率
    early_stop=20, # 早停次数 20
    metric="loss",
    loss="weighted_mse", # 损失函数：“mse";"IC';"weighted_mse"
    optimizer="adam", # 优化器
    GPU=0, # GPU编号
    n_jobs=1, # 最大并发进程数
    seed=42, # 随机数种子
    save_path=path['results']+'model', # 模型存储路径
    pretrained=None,
    )


#%% 拟合
model.fit(ds)
model.to_pickle()


#%% 预测
pred = model.predict(ds)
pred.to_csv(path['results']+'pred.csv')


#%% 绘制损失函数监控训练进程
utils.visualize_evals_result(model,save_path=path['results']+'evals_result.png')
utils.visualize_ic(pred, ds.prepare('test').iloc[:,-1],save_path=path['results']+'rankic.png')


