#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:41:21 2018

@author: wenting
"""
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Sequential
from utils import *
from preprocess import *
from hw2p2 import *
from cmudl.hw2p2.submit import *
#%%




hidden_layer=[32,64,128,256]
parts=range(1,3)
lr=0.0001
max_len=14000
batch_size=5
nepoch=40
training_routine(hidden_layer,parts,lr,max_len,batch_size,nepoch)
















#temp=np.random.rand(len(trial),nspeakers)
#enrolf=np.random.rand(len(enrol),nspeakers)
#enrolf=enrolf/enrolf.sum(axis=1)[:,np.newaxis]
#testf=np.random.rand(len(test),nspeakers)
#testf=testf/testf.sum(axis=1)[:,np.newaxis]
#score=np.zeros(len(trial))
#left=torch.tensor(enrolf[trial[:,0]])
#right=torch.tensor(testf[trial[:,1]])
#
#score=nn.functional.cosine_similarity(left,right)
#np.save('scores.npy',score.numpy())
#
#filepath='scores.npy'
#submit(filepath, outpath='submission.csv')
