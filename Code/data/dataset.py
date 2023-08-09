# -*- coding: utf-8 -*-
"""
【华泰金工】日频多因子系统：dataset

DatasetH: 数据集句柄

pivot_factor: 将单因子矩阵由一维转换为二维
melt_factor: 将单因子矩阵由二维转换为一维

linear_regression: 多元线性回归

dropna: 剔除指定列缺失样本
winsorize: 中位数去极值
neutralize: 对风格因子进行中性化
normalize: zscore标准化

"""


import numpy as np
import pandas as pd

import datetime


#%% 数据句柄
class DatasetH():
    '''
    数据集句柄
    '''
    
    def __init__(self,data,factors,labels=None,style_factors=None,leaky=0):
        '''
        根据因子和标签创建dataframe
    
        Parameters
        ----------
        data : pandas.DataFrame
            因子数据，index为datetime和instrument组成的双重索引，columns为factors、style_factors和labels
        factors : list
            因子列表，元素为因子名称.
        labels : dict, optional
            标签列表，元素为标签名称.
        style_factors : dict, optional
            风格/行业因子列表，元素为风格/行业因子名称.
        leaky : int, optional
            信息泄露天数.
        
        Returns
        -------
        None.
        
        '''
       
        self.data = data
        self.col_x = factors
        if labels is not None:
            self.col_y = labels
        if style_factors is not None:
            self.col_style = style_factors
            
        self.leaky = leaky
        
    def split(self,segments):
        '''
        划分训练、验证、测试集

        Parameters
        ----------
        segments : dict
            训练、验证、测试集划分参数. 如：
            config = {'train':('2018-01-01','2019-12-31'),
                      'valid':('2020-01-01','2020-12-31'),
                      'test':('2021-01-01','2021-06-30')}

        Returns
        -------
        None.

        '''
        # dataframe日期index
        idx = self.data.index.get_level_values(level='datetime')
        idx = pd.to_datetime(idx)
        # 提取训练、验证、测试集日期起始行和结束行
        self.idx_segments = {}
        for segment in segments.keys():
            tmp_leaky = 0
            # 训练、验证集结束行前移leaky日，防止信息泄露
            if segment in ['train', 'valid']:
                tmp_leaky = self.leaky
            self.idx_segments[segment] = [np.where(idx>=pd.to_datetime(segments[segment][0]))[0][0],
                                          np.where(idx<=pd.to_datetime(segments[segment][1])-datetime.timedelta(days=tmp_leaky))[0][-1]]
            self.segments = segments
            
    def get_col_set(self,col_set):
        if isinstance(col_set,str):
            if col_set=='label':
                cols = self.col_y
            elif col_set=='feature':
                cols = self.col_x            
            elif col_set=='all':
                cols = self.col_x + self.col_y
        elif isinstance(col_set,list):
            cols = col_set
        return cols
    
    def prepare(self,segment,col_set='all'):
        '''
        提取指定数据集

        Parameters
        ----------
        segment : str
            行索引，train, valid或test.
        col_set : str or list, optional
            列索引，提取数据. The default is 'all'.

        Returns
        -------
        data : pandas.DataFrame
            指定数据集.

        '''
        data = self.data
        data = data.loc[:,self.get_col_set(col_set)]
        data = data.iloc[self.idx_segments[segment][0]:self.idx_segments[segment][1]+1,:]
        return data
    
