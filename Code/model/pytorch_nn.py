# -*- coding: utf-8 -*-
"""
【华泰金工】日频多因子系统

NN模型

"""

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim


class NN():
    """NN Model

    Parameters
    ----------
    lr : float
        learning rate
    d_feat : int
        input dimensions for each time step
    metric : str
        the evaluate metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=20,
        hidden_size=64,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU="0",
        n_jobs=10,
        seed=None,
        save_path=None,
        pretrained=None,
        **kwargs
    ):
        # Set logger.
        print("NN pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.save_path = save_path
        self.pretrained = pretrained

        print(
            "NN parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nn_jobs : {}"
            "\nseed : {}"
            "\nsave_path : {}".format(
                d_feat,
                hidden_size,
                n_epochs,
                lr,
                metric,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                n_jobs,
                seed,
                save_path,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        if pretrained is None:
            self.NN_model = NNModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
            )
        else:
            f = open(pretrained+'model.pkl','rb')
            model = pickle.load(f)
            f.close()
            self.NN_model = model.NN_model
            
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.NN_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.NN_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self._fitted = False
        self.NN_model.to(self.device)

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def IC(self, pred, label):
        #print(type(pred))
        v_pred = pred - torch.mean(pred)
        v_label = label - torch.mean(label)
        loss = torch.mean(torch.sum(v_label*v_pred)/(torch.sqrt(torch.sum(v_label**2)) * torch.sqrt(torch.sum(v_pred**2))))
        return loss

    def weighted_mse_loss(self, pred, label):
        observation_dim = pred.size()[-1]
        mid_dim = int(0.5*observation_dim)
        midmid_dim = int(0.75*observation_dim)
        sort_pred, index_pred = torch.sort(pred,0,descending=True)
        sort_label, index_label = torch.sort(label, 0, descending=True)
        total = 0
        for i in range(observation_dim):
            if i<mid_dim:
                total = total + (sort_pred[i]-sort_label[i])**2
            elif i<midmid_dim:
                total = total + (sort_pred[i]-sort_label[i])**2 * 0.5
            else:
                total = total + (sort_pred[i] - sort_label[i]) ** 2 * 0.25
        loss = torch.mean(total/observation_dim)
        return loss


    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        if self.loss == "IC":
            return self.IC(pred[mask], label[mask])
        if self.loss == "weighted_mse":
            return self.weighted_mse_loss(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric == "" or self.metric == "loss":
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):

        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        self.NN_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):

            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred = self.NN_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.NN_model.parameters(), 3.0)
            self.train_optimizer.step()


    def test_epoch(self, data_x, data_y):

        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.NN_model.eval()
        
        with torch.no_grad():

            scores = []
            losses = []
    
            daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)
 
            for idx, count in zip(daily_index, daily_count):
                
                batch = slice(idx, idx + count)
                feature = torch.from_numpy(x_values[batch]).float().to(self.device)
                label = torch.from_numpy(y_values[batch]).float().to(self.device)
  
                pred = self.NN_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
    
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        verbose=True,
    ):

        dl_train = dataset.prepare('train')
        dl_valid = dataset.prepare('valid')

        x_train, y_train = dl_train.iloc[:,:-1], dl_train.iloc[:,-1]
        x_valid, y_valid = dl_valid.iloc[:,:-1], dl_valid.iloc[:,-1]
        
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        evals_result["time"] = []

        # train
        print("training...")
        self._fitted = True

        for step in range(self.n_epochs):
            print("Epoch%d:", step)
            print("training...")
            start_time = time.time()
            self.train_epoch(x_train, y_train)
            print("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            end_time = time.time()
            print("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            evals_result["time"].append(end_time-start_time)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.NN_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.NN_model.load_state_dict(best_param)
        if not self.save_path is None:
            torch.save(self.NN_model, self.save_path+'.pt')
        
        # 绘制损失函数使用
        evals_result['best_epoch'] = best_epoch
        self.evals_result = evals_result

        if not self.device == 'cpu':
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self._fitted:
             raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare('test')
        x_test = dl_test.iloc[:,:-1]

        index = x_test.index
        self.NN_model.eval()
        x_values = x_test.values
        preds = []

        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):

            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                if self.device == 'cpu':
                    pred = self.NN_model(x_batch).detach().numpy()
                else:
                    pred = self.NN_model(x_batch).detach().cpu().numpy()

            preds.append(pred)


        return pd.Series(np.concatenate(preds), index=index)

    def to_pickle(self):
        if not self.save_path is None:
            f = open(self.save_path+'.pkl','wb')
            pickle.dump(self,f)
            f.close()
            

class NNModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        
        self.encoder1 = nn.Linear(self.d_feat, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.encoder2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()


    def forward(self, x):
        
        hidden = self.encoder1(x)
        hidden = self.sigmoid(hidden)
        hidden = self.bn1(hidden)
        hidden = self.encoder2(hidden)
        hidden = self.sigmoid(hidden)
        hidden = self.bn2(hidden)

        # fc
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)

        return self.fc_out(hidden).squeeze()

