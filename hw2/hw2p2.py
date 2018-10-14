#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 20:15:23 2018

@author: wenting
"""

from utils import *
from preprocess import *
from hw2p2 import *
from cmudl.hw2p2.submit import *


import numpy as np
import torch,os,pdb,math
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import random

# =============================================================================
#%% load Dataset
# =============================================================================
class CNNdt(Dataset):
    """dataset for hw2p2"""
    def __init__(self, x, y=None, max_len=14000,batch_size=16, cuda=False, sorted=True): 
        self.x=x
        self.y=y
        self._n_sample = self.x.shape[0]
        self._n_batch = int(math.ceil(self._n_sample / float(batch_size)))
        self.batch_size = batch_size
        self.random_idx = range(self._n_batch) # [0, 1, ..., n_batch-1]
#        self.cuda = cuda 
        self.train = (y is not None)
        self.max_len=max_len

    def shuffle(self):
        self.random_idx = np.random.permutation(self._n_batch)
    
    def pad(self, xi):
        # xi: [BS, num_seq, 64]
        _xi = []
        for xij in xi:
            rid = random.randint(0, xij.shape[0])
            _xij = np.lib.pad(xij, ((0, self.max_len), (0,0)), 'wrap')
            _xi.append(_xij[rid: rid+self.max_len])
            
            
        return torch.FloatTensor(_xi)

    def __getitem__(self,index):
        if index == self._n_batch:
            raise StopIteration()
        s = self.random_idx[index] * self.batch_size #start point
        e = min(s+self.batch_size, self._n_sample) #end point

        if self.train:
            label = torch.LongTensor(self.y[s: e])
        else:
            label = None
        x=self.pad(self.x[s:e])[:,np.newaxis]    
#        pdb.set_trace()
        return {'feature': x, 'label': label}
  
    def __len__(self):
        return self._n_sample

# =============================================================================
#%% network model
# =============================================================================   
#the residual block code is based on:
#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
# this is not mentioned in class so I have to get some help        

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.elu = nn.ELU()

        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
#        pdb.set_trace()
        out += residual
        out = self.elu(out)
        return out        
    
    
    
class Model(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __init__(self, hidden_layer,nspeakers):
        super(Model, self).__init__()
        self.nn = nn.Sequential(
#            nn.Dropout(p=0.2),
            nn.Conv2d(1, hidden_layer[0], kernel_size=5, stride=2, bias=False),
            nn.ELU(),
            ResidualBlock(hidden_layer[0],hidden_layer[0]),
            nn.Conv2d(hidden_layer[0], hidden_layer[1], kernel_size=5, stride=2, bias=False),
            nn.ELU(),
            ResidualBlock(hidden_layer[1],hidden_layer[1]),
            nn.Conv2d(hidden_layer[1], hidden_layer[2],  kernel_size=5, stride=2, bias=False),
            nn.ELU(),
            ResidualBlock(hidden_layer[2],hidden_layer[2]),
            nn.Conv2d(hidden_layer[2], hidden_layer[3],  kernel_size=5, stride=2, bias=False),
            nn.ELU(),
            ResidualBlock(hidden_layer[3],hidden_layer[3])
            )
        self.output = nn.Linear(hidden_layer[3], nspeakers) # W * feat, W: nspeaker x 256

    def forward(self, x):
        """
        Args:
            x: [batch_size, 1,num_seq, 64]
        Return:
            pred: [batch_size, nspeakers]
        """
        feat = self.feature(x)  # [batch_size, 256]
        pred = self.output(feat)  # [batch_size, nspeakers]
        return pred

    def feature(self, x):
        """
        Args:
            x: [batch_size, num_seq, 64]
        Return:
            feature: [batch_size, 256]
        """
        bs = x.size(0)
#        x_ = x.squeeze(1)  # [bs, 1, num_seq, 64]
        feat = self.nn(x) # [bs, 256, num_seq_, 1]
        feat = feat.mean(dim=2)  # [bs, 256, 1, 1]
        feat = feat.view(bs, -1) # [bs, 256] 
        return feat

# =============================================================================
# visualizaion
# =============================================================================
def plotline(data, xlabel, ylabel, title, path):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path)
    plt.clf()


def visualize_statistics( train_accuracy,val_eer,savepath=None):
    path = os.getcwd() if savepath is None else savepath
#    plotline( avg_train_loss, "Epoch", "Loss", "Training Loss",
#             os.path.join(path, "train_loss.png"))
    plotline(train_accuracy, "Epoch", "Accuracy", "Training Accuracy",
             os.path.join(path, "train_accuracy.png"))
    plotline(val_eer, "Epoch", "Loss",
             "Validation EER", os.path.join(path, "val_eer.png"))


# =============================================================================
# training routine            
# =============================================================================


def get_features(net, data):
    feat = []
    
    for i, batch in enumerate(data):
        if net.output.weight.is_cuda:
            batch['feature']=batch['feature'].cuda()
#        pdb.set_trace()    
        y_feat = net.feature(batch['feature']).cpu().detach()
        feat.append(y_feat)
    feat = torch.cat(feat, dim=0) 
    return feat  # [N, 256]

def compute_trial(net, trial, enroll, test, label=None):
    net.eval()
#    pdb.set_trace()
    enroll_feat = get_features(net, enroll)  # [N, 256]
    test_feat = get_features(net, test)      # [M, 256]
    # cos-similarity    
    left = enroll_feat[trial[:,0]]
    right = test_feat[trial[:,1]]
    score = nn.functional.cosine_similarity(left,right)
    if label is not None:
        eer = EER(label, score)[0]
    else:
        eer = 0
    return score, eer  

def training_routine(hidden_layer,parts=range(1,2),lr=0.001,max_len=14000, batch_size=16,
                     nepoch=1, cuda=True):
    # load data 
    trn = train_load('./data', parts)
    nspeakers = trn[2]
    trn = CNNdt(x=trn[0], y=trn[1],max_len=max_len, batch_size=batch_size, 
                cuda=False, sorted=True)
    dev = dev_load('./data/dev.preprocessed.npz')
    dev_enroll = CNNdt(x=dev[2],max_len=max_len,batch_size=batch_size)
    dev_test = CNNdt(x=dev[3],max_len=max_len,batch_size=batch_size)
    tst = test_load('./data/test.preprocessed.npz')
    tst_enroll = CNNdt(x=tst[1],max_len=max_len,batch_size=batch_size)
    tst_test = CNNdt(x=tst[2],max_len=max_len,batch_size=batch_size)

    net = Model(hidden_layer, nspeakers)
    criterion = nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=0.001)
    #always train in gpu    
    if cuda:
        net = net.cuda()

    #inference initialization
    train_accuracy=np.zeros(nepoch)
#    avg_train_loss=np.zeros(nepoch)
    dev_eer=np.zeros(nepoch)
    
    log_inter = 100
    tic()
    for epoch in range(nepoch):
#        pdb.set_trace()
#        if 4<=epoch<=6:
#            for group in optimizer.param_groups: 
#                group['lr']=group['lr']*0.1
#        print('epoch:{}'.format(epoch))
        # tic()
        net.train()
        trn.shuffle()
        cnt = 0.0
        for i, batch in enumerate(trn):
            if cuda:
               batch['feature'] =batch['feature'].cuda()
               batch['label'] =batch['label'].cuda()
            y_pred = net(batch['feature']) 
            loss = criterion(y_pred, batch['label'])
            optimizer.zero_grad()
#            pdb.set_trace()
            loss.backward()
            optimizer.step()
            train_prediction = y_pred.cpu().detach().argmax(dim=1)
            train_accuracy[epoch]+= int((train_prediction==batch['label'].cpu()).long().sum())
            cnt += float(train_prediction.size(0))
#            if i % log_inter == 0:
#                print('Epoch %d, iter %d, training_accura %.4f' % (epoch, i, train_accuracy[epoch] / cnt))
        train_accuracy[epoch] /= float(len(trn))

        # Validation 
        dev_score, eer = compute_trial(net, dev[0], dev_enroll, dev_test, dev[1])
        print('Epoch %d, train_accuracy %.4f, dev_eer %.4f' % (epoch, train_accuracy[epoch], eer))
        dev_eer[epoch] = eer
    toc()
    # Test 
    tst_score, tst_eer = compute_trial(net, tst[0], tst_enroll, tst_test)
    np.save('scores.npy',tst_score.numpy())
    submit('scores.npy', outpath='submission.csv')    
    visualize_statistics(train_accuracy,dev_eer)

         

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        t=time.time() - startTime_for_tictoc
        if t<60:
            print ("Elapsed time is " + str(t) + " seconds.")
        else:
            print ("Elapsed time is " + str(t/60) + " minutes("+str(t) + " seconds).")
    else:
        print ("Toc: start time not set")

