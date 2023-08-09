# -*- coding: utf-8 -*-
"""
【华泰金工】日频多因子系统

模型训练相关工具函数

"""

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('seaborn')
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


#%% 绘制误差曲线
def visualize_evals_result(model, save_path=None):
    plt.figure(figsize = (12,8))
    plt.plot(model.evals_result['train'],label='train score') # 训练集score(score为loss相反数，参数metric定义)
    plt.plot(model.evals_result['valid'],label='valid score') # 验证集score
    
    best_epoch = model.evals_result['best_epoch']
    plt.scatter(best_epoch,model.evals_result['valid'][best_epoch],label='best epoch') # 最优迭代次数
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    if not save_path is None:
        plt.savefig(save_path, dpi=300)
   
    
#%% 计算IC
def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False):
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
    ric = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


#%% 绘制累计Rank IC
def visualize_ic(pred, label, save_path=None):
    ic, ric = calc_ic(pred, label, dropna=True)
    cum_ric = ric.cumsum()
    
    plt.figure(figsize = (12,8))
    plt.plot(pd.to_datetime(cum_ric.index),cum_ric)
    plt.title('average Rank IC = %.4f'%ric.mean())

    # 配置横坐标为日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=90)
    
    if not save_path is None:
        plt.savefig(save_path, dpi=300)

