import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr.envs import VecPyTorch

from procgen import ProcgenEnv
import argparse

import sys
sys.path.append(os.getcwd())
import pickle
from a2c_ppo_acktr.utils import init
from utils import myutils

from collections import OrderedDict

def assign_gpu_device(myargs):
    print (f"save_name: {myargs.save_name}")
    print (f"env_name: {myargs.env_name}")

    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    # print (f"******************** myargs.cuda_num: {myargs.cuda_num}")
    # print (f"******************** myargs.cuda: {myargs.cuda}")
    try:
        CUDA_VISIBLE_DEVICES_raw_str = os.environ["CUDA_VISIBLE_DEVICES"]
    except Exception as e:
        CUDA_VISIBLE_DEVICES_raw_str = "none"

    # print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_raw_str}")
    # print(f"torch.cuda.device_count():  {torch.cuda.device_count()}") 
    
    # CUDA_VISIBLE_DEVICES_raw_str_nvidia = os.environ["NVIDIA_VISIBLE_DEVICES"]
    # print(f"NVIDIA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_raw_str_nvidia}")
    if not myargs.cuda_num:
        CUDA_VISIBLE_DEVICES_raw_str_splitted = CUDA_VISIBLE_DEVICES_raw_str.split(",")
        CUDA_VISIBLE_DEVICES_raw_str_splitted_list_int = [int(item) for item in CUDA_VISIBLE_DEVICES_raw_str_splitted]
        cuda_num = f"cuda" #{CUDA_VISIBLE_DEVICES_raw_str_splitted_list_int[0]}
        # print(f"No cuda_num provided, cuda_num set to what slurm sets NVIDIA_VISIBLE_DEVICES")
        # print(f"cuda_num: {cuda_num}  ********************")
        device = torch.device(cuda_num if myargs.cuda else "cpu")
    else:
        # print(f"cuda-num provided")
        device = torch.device(myargs.cuda_num if myargs.cuda else "cpu")
    print(f"device: {device}")
    return device

def grad_g_theta_calc(MDP, agent, policy, ranked_traj_dict):
    ranks = sorted(ranked_traj_dict.keys(), reverse=True)
    num_pairs = len(ranks) - 1
    grad_g_theta = torch.zeros(agent.theta_size)
    for i in range(num_pairs):
        traj_j = ranked_traj_dict[ranks[i]]
        traj_i = ranked_traj_dict[ranks[i+1]]
        grad_R_traj_j = grad_R_traj(agent, traj_j)
        grad_R_traj_i = grad_R_traj(agent, traj_i)
        return_traj_j = ranks[i]
        return_traj_i = ranks[i+1]
        grad_g_theta += -grad_R_traj_j + ((grad_R_traj_i/math.exp(return_traj_i))+(grad_R_traj_j/math.exp(return_traj_j))) / (math.exp(return_traj_i) + math.exp(return_traj_j))        
    return grad_g_theta

def grad_policy_contraint_terms_calc(MDP, policy, ranked_traj_dict, lambdas, gammas):
    # for the old fomulation
    ranks_sorted = sorted(ranked_traj_dict.keys(), reverse=True)
    grad_policy_lambda = np.zeros((MDP.n_dim, MDP.n_dim, 4))
    grad_policy_gamma = np.zeros((MDP.n_dim, MDP.n_dim, 4))


    # calculate the gradient of the policy with respect to the lambda terms
    for idx, _ in enumerate(ranks_sorted[:-1]):
        traj1 = ranked_traj_dict[ranks_sorted[idx]]
        traj2 = ranked_traj_dict[ranks_sorted[idx]]

        # traj 1 > traj2
        state_action_freq = state_action_freq_two_trajs_calc(MDP, traj1, traj2)
        grad_policy_lambda += lambdas[idx] * np.divide(state_action_freq, policy)

    # calculate the gradient of the policy w.r.t the equality constraints gammas
    for i in range(4):
        grad_policy_gamma[:,:,i] = gammas

    return grad_policy_lambda, grad_policy_gamma

def state_action_freq_two_trajs_calc(MDP, traj1, traj2):
    state_action_freq = np.zeros((MDP.n_dim, MDP.n_dim, 4))

    for transition in traj1:
        state = transition[0]
        i,j = state[0], state[1]
        action = transition[1]
        state_action_freq[i, j, action] += 1

    for transition in traj2:
        state = transition[0]
        i,j = state[0], state[1]
        action = transition[1]
        state_action_freq[i, j, action] -= 1

    return state_action_freq

def state_action_freq_calc(MDP, sampled_trajs):
    n = len(sampled_trajs)
    state_action_freq = np.zeros((MDP.n_dim, MDP.n_dim, 4))
    
    for traj in sampled_trajs:
        for transition in traj:
            state = transition[0]
            i,j = state[0], state[1]
            action = transition[1]
            state_action_freq[i, j, action] += 1/n

    return state_action_freq

def extract_states_from_traj(traj):
    states_all = []
    states_traj = [item for item in traj]
    states_all.extend(states_traj)

    return states_all

def subsample_demos(trajs, subsample_length, subsample_increment, horizon):
    '''
    This function breaks down the demonstration trajectories into subtrajectories. It creates a list of lists. 
    Each element of the outer list stores subtrajectories corresponding

    '''
    traj_list, returns = cut_trajs_calc_return_no_device(trajs, horizon)
    max_len_trajs = max([len(traj) for traj in traj_list])
    subsampled_listOflists = []
    subsampled_listOflists_returns = []
    begin_idx = 0
    while begin_idx < max_len_trajs - 1:
        end_idx = begin_idx + subsample_length
        subsampled_list = []
        subsampled_list_returns = []
        for traj, rank in zip(traj_list, returns):
            if end_idx <= len(traj) - 1:
                subsampled_list.append(traj[begin_idx:end_idx])
                subsampled_list_returns.append(rank)

        if len(subsampled_list) > 1:
            subsampled_listOflists.append(subsampled_list)
            subsampled_listOflists_returns.append(subsampled_list_returns)
        begin_idx += subsample_increment
    
    # list(filter(None, subsampled_list_raw)) 
    return subsampled_listOflists, subsampled_listOflists_returns

def subsample_demos_true_return(trajs, subsample_length, subsample_increment, horizon):
    '''
    This function breaks down the demonstration trajectories into subtrajectories. It creates a list of lists. 
    Each element of the outer list stores subtrajectories corresponding

    '''
    traj_list = trajs

    max_len_trajs = max([len(traj) for traj in traj_list])
    subsampled_listOflists = []
    subsampled_listOflists_returns = []
    begin_idx = 0
    while begin_idx < max_len_trajs - 1:
        end_idx = begin_idx + subsample_length
        subsampled_list = []
        subsampled_list_returns = []
        for traj in traj_list:
            if end_idx <= len(traj) - 1:
                subsampled_list.append(traj[begin_idx:end_idx])
                return_subtraj = calc_traj_return(traj[begin_idx:end_idx])
                subsampled_list_returns.append(return_subtraj)

        if len(subsampled_list) > 1:
            subsampled_listOflists.append(subsampled_list)
            subsampled_listOflists_returns.append(subsampled_list_returns)
        begin_idx += subsample_increment
    
    # list(filter(None, subsampled_list_raw)) 
    return subsampled_listOflists, subsampled_listOflists_returns

def subsample_trajs(trajs, subsample_length, subsample_increment, horizon):
    '''
    This function breaks down the demonstration trajectories into subtrajectories. It creates a list of lists. 
    Each element of the outer list stores subtrajectories corresponding

    '''
    traj_list, returns = cut_trajs_calc_return_no_device(trajs, horizon)
    max_len_trajs = max([len(traj) for traj in traj_list])
    subsampled_listOflists = []
    subsampled_listOflists_returns = []
    begin_idx = 0
    while begin_idx < max_len_trajs - 1:
        end_idx = begin_idx + subsample_length
        subsampled_list = []
        subsampled_list_returns = []
        for traj, rank in zip(traj_list, returns):
            if len(traj) - 1 >= end_idx:
                subsampled_list.append(traj[begin_idx:end_idx])
                subsampled_list_returns.append(rank)

        if len(subsampled_list) > 1:
            subsampled_listOflists.append(subsampled_list)
            subsampled_listOflists_returns.append(subsampled_list_returns)
        begin_idx = end_idx
    
    # list(filter(None, subsampled_list_raw)) 
    return subsampled_listOflists, subsampled_listOflists_returns

def create_pairs(initial_trajs, initial_trajs_returns, num_pairs):
    # create pairs using fix step size subsamplint
    num_init_trajs = len(initial_trajs)
    pairs = []
    returns = []

    #add full trajs (for use on Enduro)
    for _ in range(num_pairs):

        # sample two different trajectories
        idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)

        #create random partial trajs by finding random start frame and random  frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        step = np.random.randint(3,7)
        
        traj_i = initial_trajs[idx_i][si::step]  
        traj_j = initial_trajs[idx_j][sj::step]

        traj_i_return = initial_trajs_returns[idx_i]
        traj_j_return = initial_trajs_returns[idx_j]
        
        pairs.append((traj_i, traj_j))
        returns.append((traj_i_return, traj_j_return))

    return pairs, returns

def create_pairs_no_step(initial_trajs, initial_trajs_returns, num_pairs, priority_sampling):
    # create pairs using completely random subsampling
    # All returned pairs have different returns or empty lists are returned
    num_init_trajs = len(initial_trajs)
    pairs = []
    returns = []

    initial_trajs_returns_unique = []
    initial_trajs_returns_unique_set = set()
    initial_trajs_unique = []

    for idx, ret in enumerate(initial_trajs_returns):
        if ret not in initial_trajs_returns_unique_set:
            initial_trajs_returns_unique_set.add(ret)
            initial_trajs_returns_unique.append(ret)
            initial_trajs_unique.append(initial_trajs[idx])
 
    if len(initial_trajs_returns_unique_set) < 2:
        pass
    else:
        for _ in range(num_pairs):
            # sample two different trajectories
            traj_i_return, traj_j_return = 0, 0
            while traj_i_return == traj_j_return:
                if priority_sampling:
                    idx_i, idx_j = 0, 0
                    while idx_i == idx_j:
                        initial_trajs_returns_shifted = initial_trajs_returns + abs(min(initial_trajs_returns)) 
                        probabilities = initial_trajs_returns_shifted / sum(initial_trajs_returns_shifted)
                        idx_i = np.random.choice(num_init_trajs, size=1, p=probabilities)[0]
                        idx_j = np.random.choice(num_init_trajs, size=1)[0]
                else:
                    idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)
                traj_i_return = initial_trajs_returns[idx_i]
                traj_j_return = initial_trajs_returns[idx_j]

            #create random partial trajs by randomly subsampling the two pairs
            subsample_ratio = np.random.randint(3,7)
            num_idxs_traj_i = int(len(initial_trajs[idx_i]) / subsample_ratio)
            num_idxs_traj_j = int(len(initial_trajs[idx_j]) / subsample_ratio)

            traj_i_idxs = sorted(np.random.choice(len(initial_trajs[idx_i]), size=num_idxs_traj_i, replace=False).tolist())
            traj_j_idxs = sorted(np.random.choice(len(initial_trajs[idx_j]), size=num_idxs_traj_j, replace=False).tolist())

            traj_i = [initial_trajs[idx_i][idx] for idx in traj_i_idxs]
            traj_j = [initial_trajs[idx_j][idx] for idx in traj_j_idxs]
            
            pairs.append((traj_i, traj_j))
            returns.append((traj_i_return, traj_j_return))
    return pairs, returns

def create_pairs_no_step_no_subsample(initial_trajs, initial_trajs_returns, num_pairs, priority_sampling):
    # create pairs using completely random subsampling
    # All returned pairs have different returns or empty lists are returned
    num_init_trajs = len(initial_trajs)
    pairs = []
    returns = []

    initial_trajs_returns_unique = []
    initial_trajs_returns_unique_set = set()
    initial_trajs_unique = []

    for idx, ret in enumerate(initial_trajs_returns):
        if ret not in initial_trajs_returns_unique_set:
            initial_trajs_returns_unique_set.add(ret)
            initial_trajs_returns_unique.append(ret)
            initial_trajs_unique.append(initial_trajs[idx])
 
    if len(initial_trajs_returns_unique_set) < 2:
        pass
    else:
        for _ in range(num_pairs):
            # sample two different trajectories
            traj_i_return, traj_j_return = 0, 0
            while traj_i_return == traj_j_return:
                if priority_sampling:
                    idx_i, idx_j = 0, 0
                    while idx_i == idx_j:
                        initial_trajs_returns_shifted = initial_trajs_returns + abs(min(initial_trajs_returns)) 
                        probabilities = initial_trajs_returns_shifted / sum(initial_trajs_returns_shifted)
                        idx_i = np.random.choice(num_init_trajs, size=1, p=probabilities)[0]
                        idx_j = np.random.choice(num_init_trajs, size=1)[0]
                else:
                    idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)
                traj_i_return = initial_trajs_returns[idx_i]
                traj_j_return = initial_trajs_returns[idx_j]


            traj_i = initial_trajs[idx_i]
            traj_j = initial_trajs[idx_j]
            
            pairs.append((traj_i, traj_j))
            returns.append((traj_i_return, traj_j_return))
    return pairs, returns

def create_pairs_no_step_std_apart(initial_trajs, initial_trajs_returns, num_pairs, priority_sampling):
    # create pairs using completely random subsampling
    # All returned pairs have different returns or empty lists are returned
    num_init_trajs = len(initial_trajs)
    pairs = []
    returns = []
    returns_std = np.std(initial_trajs_returns)

 
    if len(initial_trajs_returns) < 2:
        pass
    else:
        for _ in range(num_pairs):
            # sample two different trajectories
            traj_i_return, traj_j_return = 0, 0
            while np.abs(traj_i_return-traj_j_return) <= returns_std:
                idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)
                traj_i_return = initial_trajs_returns[idx_i]
                traj_j_return = initial_trajs_returns[idx_j]

            #create random partial trajs by randomly subsampling the two pairs
            subsample_ratio = np.random.randint(3,7)
            num_idxs_traj_i = int(len(initial_trajs[idx_i]) / subsample_ratio)
            num_idxs_traj_j = int(len(initial_trajs[idx_j]) / subsample_ratio)

            traj_i_idxs = sorted(np.random.choice(len(initial_trajs[idx_i]), size=num_idxs_traj_i, replace=False).tolist())
            traj_j_idxs = sorted(np.random.choice(len(initial_trajs[idx_j]), size=num_idxs_traj_j, replace=False).tolist())

            traj_i = [initial_trajs[idx_i][idx] for idx in traj_i_idxs]
            traj_j = [initial_trajs[idx_j][idx] for idx in traj_j_idxs]
            
            pairs.append((traj_i, traj_j))
            returns.append((traj_i_return, traj_j_return))
    return pairs, returns

def create_pairs_no_step_distance_apart_subsample(initial_trajs, initial_trajs_returns, num_pairs, priority_sampling):
    """ 
    create pairs using completely random subsampling
    All returned pairs have more than 1/10 of the range of returns difference
    """
    num_init_trajs = len(initial_trajs)
    pairs = []
    returns = []

    std_returns = np.std(initial_trajs_returns)
    max_returns = np.max(initial_trajs_returns)
    min_returns = np.min(initial_trajs_returns)
    minimum_allowed_distance = (max_returns - min_returns) / 10


    if int(std_returns) == 0:
        pass
    elif len(initial_trajs_returns) < 2:
        pass
    else:
        for _ in range(num_pairs):
            # sample two different trajectories
            traj_i_return, traj_j_return = 0, 0
            while np.abs(traj_i_return-traj_j_return) <= minimum_allowed_distance:
                idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)
                traj_i_return = initial_trajs_returns[idx_i]
                traj_j_return = initial_trajs_returns[idx_j]

            #create random partial trajs by randomly subsampling the two pairs
            subsample_ratio = np.random.randint(3,7)
            num_idxs_traj_i = int(len(initial_trajs[idx_i]) / subsample_ratio)
            num_idxs_traj_j = int(len(initial_trajs[idx_j]) / subsample_ratio)

            traj_i_idxs = sorted(np.random.choice(len(initial_trajs[idx_i]), size=num_idxs_traj_i, replace=False).tolist())
            traj_j_idxs = sorted(np.random.choice(len(initial_trajs[idx_j]), size=num_idxs_traj_j, replace=False).tolist())

            traj_i = [torch.tensor(initial_trajs[idx_i][idx], device=self.device) for idx in traj_i_idxs]
            traj_j = [torch.tensor(initial_trajs[idx_j][idx], device=self.device) for idx in traj_j_idxs]
            
            pairs.append((traj_i, traj_j))
            returns.append((traj_i_return, traj_j_return))
    return pairs, returns

def create_pairs_distance_apart_device_hardDrive(save_path_new_trajs, initial_trajs_returns, num_pairs, priority_sampling, device):
    """ 
    create pairs using no subsampling by reading trajs from hard drive
    All returned pairs have more than 1/10 of the range of returns difference
    """
    num_init_trajs = len(initial_trajs_returns)
    pairs = []
    returns = []

    std_returns = np.std(initial_trajs_returns)
    max_returns = np.max(initial_trajs_returns)
    min_returns = np.min(initial_trajs_returns)
    minimum_allowed_distance = (max_returns - min_returns) / 10


    if std_returns < 0.00000001:
        pass
    elif len(initial_trajs_returns) < 2:
        pass
    else:
        for _ in range(num_pairs):
            # sample two different trajectories
            traj_i_return, traj_j_return = 0, 0
            while np.abs(traj_i_return-traj_j_return) <= minimum_allowed_distance:
                idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)
                traj_i_return = initial_trajs_returns[idx_i]
                traj_j_return = initial_trajs_returns[idx_j]
            traj_i = torch.load(save_path_new_trajs+f"/traj_{idx_i}.pt", map_location=device)
            traj_j = torch.load(save_path_new_trajs+f"/traj_{idx_j}.pt", map_location=device)

            # traj_i = [torch.tensor(obs.clone(), device=device) for obs in initial_trajs[idx_i]]
            # traj_j = [torch.tensor(obs.clone(), device=device) for obs in initial_trajs[idx_j]]

            # traj_i = [obs.cuda(device=device) for obs in traj_i]
            # traj_j = [obs.cuda(device=device) for obs in traj_j]

            pairs.append((traj_i, traj_j))
            returns.append((traj_i_return, traj_j_return))

    return pairs, returns

def create_pairs_distance_apart_device_memory(initial_trajs, initial_trajs_returns, num_pairs, priority_sampling, device):
    """ 
    create pairs using no subsampling, assuming trajs are stored in memory
    All returned pairs have more than 1/10 of the range of returns difference
    """
    num_init_trajs = len(initial_trajs)
    pairs = []
    returns = []

    std_returns = np.std(initial_trajs_returns)
    max_returns = np.max(initial_trajs_returns)
    min_returns = np.min(initial_trajs_returns)
    minimum_allowed_distance = (max_returns - min_returns) / 10


    if int(std_returns) == 0:
        pass
    elif len(initial_trajs_returns) < 2:
        pass
    else:
        for _ in range(num_pairs):
            # sample two different trajectories
            traj_i_return, traj_j_return = 0, 0
            while np.abs(traj_i_return-traj_j_return) <= minimum_allowed_distance:
                idx_i, idx_j = np.random.choice(num_init_trajs, size=2, replace=False)
                traj_i_return = initial_trajs_returns[idx_i]
                traj_j_return = initial_trajs_returns[idx_j]

            # traj_i = [torch.tensor(obs.clone(), device=device) for obs in initial_trajs[idx_i]]
            # traj_j = [torch.tensor(obs.clone(), device=device) for obs in initial_trajs[idx_j]]

            traj_i = [obs.cuda(device=device) for obs in initial_trajs[idx_i]]
            traj_j = [obs.cuda(device=device) for obs in initial_trajs[idx_j]]
            pairs.append((traj_i, traj_j))
            returns.append((traj_i_return, traj_j_return))

    return pairs, returns

def calc_traj_return(traj):
    return np.sum([transition[-1].item() for transition in traj])

def cut_trajs_calc_return(trajs, horizon, device):
    trajs_cut = [traj[:horizon] for traj in trajs]
    traj_list = []
    returns = []
    for traj in trajs_cut:
        traj_return = np.sum([transition[-1].item() for transition in traj])
        traj_corret_device = [[item[0].to(device), item[1]] for item in traj]
        if traj_return not in returns:
            traj_list.append(traj_corret_device)
            returns.append(traj_return)

    # traj_list = [x for _,x in sorted(zip(returns, traj_list), reverse=True)]
    # returns = sorted(returns, reverse=True)

    # ranks = sorted(ranked_traj_dict.keys(), reverse=True)
    return traj_list, returns

def cut_trajs_calc_return_no_device(trajs, horizon):
    trajs_cut = [traj[:horizon] for traj in trajs]
    traj_list = []
    returns = []
    for traj in trajs_cut:
        traj_return = np.sum([transition[-1].item() for transition in traj])
        if traj_return not in returns:
            traj_list.append(traj)
            returns.append(traj_return)

    # traj_list = [x for _,x in sorted(zip(returns, traj_list), reverse=True)]
    # returns = sorted(returns, reverse=True)

    # ranks = sorted(ranked_traj_dict.keys(), reverse=True)
    return traj_list, returns

def trajs_calc_return_no_device(trajs, discounted_rew, gamma):
    """
    calculates the undiscounted return of trajectories. Returns a list of  trajectories 
    and a list of corresponding returns separately
    """
    trajs_cut = trajs
    traj_list = []
    returns = []
    for traj in trajs_cut:
        if discounted_rew:
            traj_len = len(traj)
            weights = [gamma**idx for idx in range(traj_len)]
            rewards = [transition[-1].item() for transition in traj]
            traj_return = np.sum([weight*reward for (weight, reward) in zip(weights, rewards)])
        else:
            traj_return = np.sum([transition[-1].item() for transition in traj]) 

        traj_pure = [transition[0] for transition in traj]
        traj_list.append(traj_pure)
        returns.append(traj_return)

    # traj_list = [x for _,x in sorted(zip(returns, traj_list), reverse=True)]
    # returns = sorted(returns, reverse=True)

    # ranks = sorted(ranked_traj_dict.keys(), reverse=True)
    return traj_list, returns

