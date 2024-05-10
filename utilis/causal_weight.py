import os
import pickle
import numpy as np
import pandas as pd
import time
from causallearnmain.causallearn.search.FCMBased import lingam
import sys
from utilis.config import ARGConfig
import torch
import torch.nn.functional as F
import ipdb

def get_sa2r_weight(env, memory, agent, sample_size=5000, causal_method='DirectLiNGAM'):
    
    states, actions, rewards, next_states, dones = memory.sample(sample_size)
    rewards = np.squeeze(rewards[:sample_size]) 
    rewards = np.reshape(rewards, (sample_size, 1))
    X_ori = np.hstack((states[:sample_size,:], actions[:sample_size,:], rewards)) 
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))
    
    if causal_method=='DirectLiNGAM':
        start_time = time.time()  
        model = lingam.DirectLiNGAM()
        model.fit(X)
        end_time = time.time()
        model._running_time = end_time - start_time
        weight_r = model.adjacency_matrix_[-1,np.shape(states)[1]:(np.shape(states)[1]+np.shape(actions)[1])]

    #softmax weight_r
    weight = F.softmax(torch.Tensor(weight_r),0)
    weight = weight.numpy()   
    #* multiply by action size
    weight = weight * weight.shape[0]
    return weight, model._running_time

