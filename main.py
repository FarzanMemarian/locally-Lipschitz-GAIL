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
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SumxmaryWriter


from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail, gail_lipschitz, ppo_lipschitz
import a2c_ppo_acktr.arguments as arguments #import get_args, get_init, get_init_Lipschitz
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr.envs import VecPyTorch
from baselines.common.atari_wrappers import FrameStack, ClipRewardEnv, WarpFrame

from procgen import ProcgenEnv
import argparse

import sys
sys.path.append(os.getcwd())
import pickle
from a2c_ppo_acktr.utils import init
from utils import myutils
import mujoco_py

from os import listdir
from os.path import isfile, join
import math
import psutil
import tracemalloc
import linecache
# from memory_profiler import profile
# breakpoint()


class net_MLP(nn.Module):

    def __init__(self, 
                input_size,
                rew_sign,
                rew_mag,
                FC1_dim = 256,
                FC2_dim = 256,
                FC3_dim = 256,  
                out_dim=1):
        super().__init__()
        # an affine operation: y = Wx + b
        
        self.n_dim = input_size
        self.fc1 = nn.Linear(self.n_dim, FC1_dim)
        self.fc2 = nn.Linear(FC1_dim, FC2_dim)
        self.fc3 = nn.Linear(FC2_dim, FC3_dim)
        self.fc4 = nn.Linear(FC3_dim, out_dim)

        self.rew_sign = rew_sign
        self.rew_mag = rew_mag

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.rew_sign == "free":
            x = self.fc4(x)
        elif self.rew_sign == "neg":
            x = - F.relu(self.fc4(x))
        elif self.rew_sign == "pos":
            x =  F.relu(self.fc4(x))
        elif self.rew_sign == "pos_sigmoid":
            x =  self.rew_mag*torch.sigmoid(self.fc4(x))
        elif self.rew_sign == "neg_sigmoid":
            x =  - self.rew_mag*torch.sigmoid(self.fc4(x))
        elif self.rew_sign == "tanh":
            x =  self.rew_mag*torch.tanh(self.fc4(x))
        return x

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

class net_CNN(nn.Module):
    def __init__(self, 
            observation_space_shape, 
            rew_sign, 
            rew_mag,
            final_conv_channels=10):
        super().__init__()
        depth = observation_space_shape[0]
        n_dim = observation_space_shape[1]
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.rew_sign = rew_sign
        self.rew_mag = rew_mag

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        conv1_output_width = conv2d_size_out(n_dim, 8,4)
        conv2_output_width = conv2d_size_out(conv1_output_width, 4,2)
        conv3_output_width = conv2d_size_out(conv2_output_width, 3,1)
        conv4_output_width = conv2d_size_out(conv3_output_width, 7,1)
        FC_input_size = conv4_output_width * conv4_output_width * final_conv_channels
        
        self.conv1 = nn.Conv2d(depth, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, final_conv_channels, 7, stride=1)
        self.FC =    nn.Linear(FC_input_size, 1)
        # self.main = nn.Sequential(
        #             nn.Conv2d(n_dim, 32, 8, stride=4), nn.ReLU(),
        #             nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        #             nn.Conv2d(64, 66, 3, stride=1), nn.ReLU(), 
        #             nn.Conv2d(64, final_conv_channels, 7, stride=1), nn.ReLU(), 
        #             Flatten(),
        #             nn.Linear(FC_input_size, 1)
        #             )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # flatten
        if self.rew_sign == "free":
            x = self.FC(x)
        elif self.rew_sign == "neg":
            x = - F.relu(self.FC(x))
        elif self.rew_sign == "pos":
            x =  F.relu(self.FC(x))
        elif self.rew_sign == "pos_sigmoid":
            x =  self.rew_mag*torch.sigmoid(self.FC(x))
        elif self.rew_sign == "neg_sigmoid":
            x =  - self.rew_mag*torch.sigmoid(self.FC(x))
        elif self.rew_sign == "tanh":
            x =  self.rew_mag*torch.tanh(self.FC(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    


# Auxiliary functions

class reward_cl():

    def __init__(self, device, observation_space_shape, lr, rew_sign, rew_mag, rew_kwargs):

        # new networks
        self.device = device

        if len(observation_space_shape) == 1:
            # num_stacked_obs = 4
            self.reward_net = net_MLP(observation_space_shape[0], rew_sign, rew_mag, FC1_dim=rew_kwargs['FC_dim'], FC2_dim=rew_kwargs['FC_dim'], out_dim=1).to(self.device)
        elif len(observation_space_shape) == 3:
            self.reward_net = net_CNN(observation_space_shape, rew_sign, rew_mag).to(self.device)
        
        # NOTE: by commenting the following lines, we rely on Pytorch's initialization. 
        # Pytorch uses Kaiming Initialization which is good for linear layers with ReLu activations
        # self.init_weights_var = 0.05
        # self._init_weights(self.reward_net) 
        # 

        self.lr = lr
        # create the optimizer
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.001, amsgrad=False)

        # theta based variables
        self.reward_input_batch = None
        self.theta_size = self._get_size_theta()
        # self.grad_R_theta = np.zeros((self.MDP.n_dim, self.theta_size))

    def _get_size_theta(self):
        size = 0
        for f in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            dims = f.size()
            layer_size = 1
            for dim in dims:
                layer_size *= dim
            size += layer_size
        return size
 
    def _init_weights(self, reward_net):
        with torch.no_grad():
            for layer_w in reward_net.parameters():
                torch.nn.init.normal_(layer_w, mean=0.0, std=self.init_weights_var)
                # torch.nn.init.xavier_normal_(layer_w, gain=1.0)

    def reward_net_input_method(self,obs):
        return obs

    def reward_net_input_batch_traj_method(self, traj):
        reward_input_batch = torch.cat([torch.unsqueeze(trans, dim=0) for trans in traj], dim=0)
        # reward_input_batch.requires_grad_(True)
        return reward_input_batch

    def reward_net_input_batch_traj_method_stacked(self, traj):
        stacked_obs_list = []
        for idx in range(len(traj)):
            if idx == 0:
                stacked_obs = torch.cat([traj[0][0], traj[0][0], traj[0][0], traj[0][0]], dim=1) 
            elif idx == 1:
                stacked_obs = torch.cat([traj[1][0], traj[0][0], traj[0][0], traj[0][0]], dim=1) 
            elif idx == 2:
                stacked_obs = torch.cat([traj[2][0], traj[1][0], traj[0][0], traj[0][0]], dim=1) 
            else:
                stacked_obs = torch.cat([traj[idx][0], traj[idx-1][0], traj[idx-2][0], traj[idx-3][0]], dim=1)
            stacked_obs_list.append(stacked_obs)
        return torch.cat(stacked_obs_list, dim=0)

    def _get_flat_grad(self):
        # this part is to get the thetas to be used for l2 regularization
        # grads_flat = torch.zeros(self.theta_size)
        grads_flat_list = []
        start_pos = 0
        for idx, weights in enumerate(self.reward_net.parameters()):
            # SHOULD BE DOUBLE CHECKED 1234567891011
            num_flat_features = self._num_flat_features(weights)

            try:
                grads = copy.deepcopy(weights.grad.view(-1, num_flat_features))      
            except Exception as e:
                print("No gradient error")

            # grads_flat[start_pos:start_pos+num_flat_features] = grads[:]
            # start_pos += num_flat_features
            grads_flat_list.append(grads)
        grads_flat = torch.unsqueeze(torch.cat(grads_flat_list, dim=1), dim=0)
        return grads_flat

    def get_flat_weights(self):
        # this part is to get the thetas to be used for l2 regularization
        weights_flat = torch.zeros(self.theta_size)
        start_pos = 0
        for weights in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            num_flat_features = self._num_flat_features(weights)
            weights = copy.deepcopy(weights.view(-1, num_flat_features).detach())
            weights_flat[start_pos:start_pos+num_flat_features] = weights[:]
            start_pos += num_flat_features
        return weights_flat

    def _num_flat_features(self, x):
        size = x.size()  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

    def _get_size_theta(self):
        size = 0
        for f in self.reward_net.parameters():
            # SHOULD BE DOUBLE CHECKED 1234567891011
            dims = f.size()
            layer_size = 1
            for dim in dims:
                layer_size *= dim
            size += layer_size
        return size


class train_GAIL():


    def __init__(self, kwargs, myargs):

        # if myargs.G_lip or myargs.D_lip:
        #     if myargs.pert_radius: 
        #         myargs.pert_radius = float(myargs.pert_radius)
        #     elif os.getenv('RADIUS'): # this means the pert_radius is not provided by the user in command line arguments, it must be read from enviornment variables
        #         myargs.pert_radius = float(os.getenv('RADIUS'))/1000
        #     else:
        #         raise ValueError('perturbation radius not provided as command line argument or as an enviornment variable')


        #     if myargs.lip_coeff: 
        #         myargs.lip_coeff = float(myargs.lip_coeff)
        #     elif os.getenv('LIP_COEFF'): # this means the lip_coeff is not provided by the user in command line arguments, it must be read from enviornment variables
        #         myargs.lip_coeff = float(os.getenv('LIP_COEFF'))/10
        #     else:
        #         raise ValueError('lip_coeff not provided as command line argument or as an enviornment variable')


        # pert_radius_list = [0.03, 0.1, 0.3]
        # pert_radius_list = [0.01, 0.03, 0.1, 0.3]

        if myargs.D_lip:
            pert_radius_list = [0.003, 0.01, 0.03, 0.1]
        elif myargs.G_lip:
            pert_radius_list = [0.03, 0.1, 0.3, 1.0]
            # pert_radius_list = [1.0]

        # lip_coeff_list = [3.0] 
        lip_coeff_list = [0.3, 1.0, 3.0, 10.0] 


        if myargs.G_lip or myargs.D_lip:
            import itertools
            list_of_reg_params = []
            for i in itertools.product(pert_radius_list, lip_coeff_list):
                list_of_reg_params.append(i)

            if myargs.pert_radius: 
                myargs.pert_radius = float(myargs.pert_radius)
            elif os.getenv('RADIUS'): # this means the pert_radius is not provided by the user in command line arguments, it must be read from enviornment variables
                # SLURM_ARRAY_TASK_COUNT = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
                SLURM_ARRAY_TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
                # divisor_for_lip_coeff = int(SLURM_ARRAY_TASK_COUNT / len(lip_coeff_list))
                # myargs.pert_radius = pert_radius_list[int(os.getenv('RADIUS'))%len(pert_radius_list)]
                myargs.pert_radius = list_of_reg_params[SLURM_ARRAY_TASK_ID][0]
            else:
                raise ValueError('perturbation radius not provided as command line argument or as an enviornment variable')


            if myargs.lip_coeff: 
                myargs.lip_coeff = float(myargs.lip_coeff)
            elif os.getenv('LIP_COEFF'): # this means the lip_coeff is not provided by the user in command line arguments, it must be read from enviornment variables
                # myargs.lip_coeff = lip_coeff_list[int(os.getenv('LIP_COEFF'))//divisor_for_lip_coeff]
                myargs.lip_coeff = list_of_reg_params[SLURM_ARRAY_TASK_ID][1]

            else:
                raise ValueError('lip_coeff not provided as command line argument or as an enviornment variable')



        if myargs.noisy_training != "No" and not (myargs.G_lip or myargs.D_lip):
            if myargs.train_noise: 
                myargs.train_noise = float(myargs.train_noise)
            elif os.getenv('TRAIN_NOISE'): # this means the lip_coeff is not provided by the user in command line arguments, it must be read from enviornment variables
                myargs.train_noise = float(os.getenv('TRAIN_NOISE'))/1000
            else:
                raise ValueError('train_noise not provided as command line argument or as an enviornment variable')

        if myargs.noisy_training != "No" and (myargs.G_lip or myargs.D_lip):
            myargs.train_noise = float(myargs.pert_radius)


        if not myargs.seed == -12345: 
            # seed is provided as command line argument and nothing needs to be done
            pass
        else:        
            if os.getenv('SEED'):
                myargs.seed = int(os.getenv('SEED'))
            else:
                raise ValueError('SEED not provided as command line argument or as an enviornment variable')


        add_to_save_name = ""

        if myargs.save_name:
            add_to_save_name += f"-{myargs.save_name}"

        if   myargs.G_lip and not myargs.D_lip and myargs.noisy_training == "No":
            myargs.save_name = f"GAIL-{myargs.env_name}-Glip-lipCoef{myargs.lip_coeff}-radius{myargs.pert_radius}{add_to_save_name}-{myargs.lip_norm}-s{myargs.seed}" 
        
        elif myargs.D_lip and not myargs.G_lip and myargs.noisy_training == "No":
            myargs.save_name = f"GAIL-{myargs.env_name}-Dlip-lipCoef{myargs.lip_coeff}-radius{myargs.pert_radius}{add_to_save_name}-{myargs.lip_norm}-s{myargs.seed}" 
        
        elif myargs.D_lip and myargs.G_lip and myargs.noisy_training == "No":
            myargs.save_name = f"GAIL-{myargs.env_name}-Glip-Dlip-lipCoef{myargs.lip_coeff}-radius{myargs.pert_radius}{add_to_save_name}-{myargs.lip_norm}-s{myargs.seed}" 
        
        elif myargs.D_lip and myargs.noisy_training != "No" and not myargs.G_lip:
            myargs.save_name = f"GAIL-{myargs.env_name}-Dlip-lipCoef{myargs.lip_coeff}-radius{myargs.pert_radius}-trainNoise{myargs.noisy_training}{myargs.train_noise}{add_to_save_name}-{myargs.lip_norm}-s{myargs.seed}" 
        
        elif myargs.G_lip and myargs.noisy_training != "No" and not myargs.D_lip:
            myargs.save_name = f"GAIL-{myargs.env_name}-Glip-lipCoef{myargs.lip_coeff}-radius{myargs.pert_radius}-trainNoise{myargs.noisy_training}{myargs.train_noise}{add_to_save_name}-{myargs.lip_norm}-s{myargs.seed}" 
        
        elif myargs.noisy_training != "No" and not myargs.G_lip and not myargs.D_lip:
            myargs.save_name = f"GAIL-{myargs.env_name}-trainNoise{myargs.noisy_training}{myargs.train_noise}-{myargs.lip_norm}-s{myargs.seed}"
        
        else:
            myargs.save_name = f"GAIL-{myargs.env_name}{add_to_save_name}-{myargs.lip_norm}-s{myargs.seed}"

        self.kwargs, self.myargs = kwargs, myargs
        self.init_params = []
        torch.manual_seed(myargs.seed)
        torch.cuda.manual_seed_all(myargs.seed)
        np.random.seed(myargs.seed)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        self.device = myutils.assign_gpu_device(myargs)

        self.log_dir = myargs.log_dir + "/" + myargs.save_name 

        # log_dir = os.path.expanduser(myargs.log_dir)
        self.eval_log_dir = self.log_dir + "_eval"
        self.log_file_name = myargs.env_name

        utils.create_dir(self.log_dir)
        utils.create_dir(self.eval_log_dir)


        with open(self.log_dir + "/" + self.log_file_name  + ".txt", "w") as file:
            file.write("Updates , num timesteps , FPS, number of Last training episodes, dist_entropy, value_loss, action_loss, mean reward, median reward, min reward, max reward \n")

        if self.myargs.eval_interval:
            with open(self.eval_log_dir + "/" + self.log_file_name  + ".txt", "w") as file:
                file.write("num_episodes, median_reward, max_reward \n")

        # envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
        #                      myargs.gamma, myargs.log_dir, device, False)

        self.list_noise_levels = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0] #  

        self._save_attributes_and_hyperparameters()

        self.train()

    def _save_attributes_and_hyperparameters(self):
        print ("saving attributes_and_hyperparameters .....")



        with open(self.log_dir + "/init_params.txt", "w") as file:
            for key in self.init_params:
                file.write(f"{key} : {self.init_params[key]} \n" )

        args_dict = vars(self.myargs)
        with open(self.log_dir +"/args_dict.pkl", "wb") as f:
            pickle.dump(args_dict, f)
        with open(self.log_dir + "/myargs.txt", "w") as file:
            for key in args_dict:
                file.write(f"{key} : {args_dict[key]} \n" )
    
        with open(self.log_dir +"/kwargs.pkl", "wb") as f:
            pickle.dump(self.kwargs, f)    
        with open(self.log_dir + "/kwargs.txt", "w") as file:
            for key in self.kwargs:
                file.write(f"{key} : {self.kwargs[key]} \n" )

    def add_noise_to_obs(self, obs, radius, norm):
        obs_size = obs.size()
        # delta has a norm ifinity smaller than radius
        if norm == "L_inf":
            delta = ((torch.rand(obs_size).to(obs.device) - 0.5 ) * radius / 0.5)
        elif norm == "L_2":
            means = torch.zeros(obs_size)
            stds = torch.ones(obs_size)
            delta = torch.normal(means, stds).to(obs.device)
            norms = torch.unsqueeze(torch.norm(delta, dim=1), dim=1)
            norms_cat = torch.cat([norms for _ in range(obs_size[-1])], dim=1)
            delta = delta * radius / norms_cat
        else:
            raise Exception("L_p norm other than L_infinity and L_2 not implemented yet")
        obs_pertubed = obs + delta 
        return obs_pertubed

    def train(self):

        # tracemalloc.start()
        myargs, kwargs = self.myargs, self.kwargs

        envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
            myargs.gamma, self.log_dir, self.device, allow_early_resets=False,  num_frame_stack=None,  **kwargs)

        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            self.device,
            base_kwargs={'recurrent': myargs.recurrent_policy})
        actor_critic.to(self.device)

        if myargs.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                alpha=myargs.alpha,
                max_grad_norm=myargs.max_grad_norm)

        elif myargs.algo == 'ppo':
            if myargs.G_lip:
                agent = ppo_lipschitz.PPO_Lipschitz(
                    actor_critic,
                    # actor_critic_lip,
                    myargs.clip_param,
                    myargs.ppo_epoch,
                    myargs.num_mini_batch,
                    myargs.value_loss_coef,
                    myargs.entropy_coef,
                    myargs.lip_coeff,
                    myargs.pert_radius,
                    myargs.lip_norm,
                    lr=myargs.lr,
                    eps=myargs.eps,
                    max_grad_norm=myargs.max_grad_norm)
            else:
                agent = algo.PPO(
                    actor_critic,
                    myargs.clip_param,
                    myargs.ppo_epoch,
                    myargs.num_mini_batch,
                    myargs.value_loss_coef,
                    myargs.entropy_coef,
                    lr=myargs.lr,
                    eps=myargs.eps,
                    max_grad_norm=myargs.max_grad_norm)

        elif myargs.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, myargs.value_loss_coef, myargs.entropy_coef, acktr=True)

        if myargs.gail:
            assert len(envs.observation_space.shape) == 1

            if myargs.D_lip or myargs.G_lip or myargs.noisy_training != "No":
                if myargs.noisy_training != "No":
                    myargs.pert_radius = myargs.train_noise

                discr = gail_lipschitz.Discriminator(
                        envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
                        self.device, myargs.pert_radius, myargs.lip_coeff, myargs.D_lip, myargs.lip_norm)
            else:
                discr = gail.Discriminator(
                    envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
                    self.device)

            file_name = os.path.join(
                myargs.gail_experts_dir, "trajs_{}.pt".format(
                    myargs.env_name.split('-')[0].lower()))
            
            expert_dataset = gail.ExpertDataset(
                file_name, myargs.noisy_training, myargs.train_noise, myargs.lip_norm, num_trajectories=4, subsample_frequency=20)
            drop_last = len(expert_dataset) > myargs.gail_batch_size
            gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=myargs.gail_batch_size,
                shuffle=True,
                drop_last=drop_last)

        rollouts = RolloutStorage(myargs.num_steps, myargs.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

        obs = envs.reset()
        if myargs.noisy_training in ["G", "DG"]:
            obs = self.add_noise_to_obs(obs, myargs.train_noise, myargs.lip_norm)
        rollouts.obs[0].copy_(obs)
        rollouts.to(self.device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(
            myargs.num_env_steps) // myargs.num_steps // myargs.num_processes

        eval_episode_rewards_list_all = []
        for j in range(num_updates):
            # if myargs.verbose:
            print(f"num overal updates: {j} ********** ")

            if myargs.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)

            for step in range(myargs.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                if myargs.noisy_training in ["G", "DG"]:
                    obs = self.add_noise_to_obs(obs, myargs.train_noise, myargs.lip_norm)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            # Here the discriminator is updated and the rewards for each transision are replaced 
            # by the discriminator's predictions
            disc_loss_lip_avg = 0.0
            disc_loss_expert_avg = 0.0 
            disc_loss_policy_avg = 0.0
            disc_loss_total_avg = 0.0
            acc_expert_avg = 0 
            acc_expert_pert_avg = 0
            acc_policy_avg = 0
            acc_policy_pert_avg = 0

            if myargs.gail:
                if j >= 10:
                    envs.venv.eval()

                gail_epoch = myargs.gail_epoch
                if j < 10:
                    gail_epoch = 100  # Warm up

                for _ in range(gail_epoch):
                    (disc_loss_total, disc_loss_lip, disc_loss_expert, disc_loss_policy, 
                        acc_expert, acc_expert_pert, acc_policy, acc_policy_pert)= discr.update(gail_train_loader, 
                        rollouts, utils.get_vec_normalize(envs)._obfilt)

                    disc_loss_total_avg += disc_loss_total
                    disc_loss_lip_avg += disc_loss_lip
                    disc_loss_expert_avg += disc_loss_expert
                    disc_loss_policy_avg += disc_loss_policy
                    acc_expert_avg += acc_expert
                    acc_expert_pert_avg += acc_expert_pert
                    acc_policy_avg += acc_policy
                    acc_policy_pert_avg += acc_policy_pert

                for step in range(myargs.num_steps):
                    rollouts.rewards[step] = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], myargs.gamma,
                        rollouts.masks[step])
                disc_loss_total_avg /= gail_epoch
                disc_loss_lip_avg /= gail_epoch
                disc_loss_expert_avg /= gail_epoch
                disc_loss_policy_avg /= gail_epoch
                acc_expert_avg /= gail_epoch 
                acc_expert_pert_avg /= gail_epoch
                acc_policy_avg /= gail_epoch
                acc_policy_pert_avg /= gail_epoch



            rollouts.compute_returns(next_value, myargs.use_gae, myargs.gamma,
                                     myargs.gae_lambda, myargs.use_proper_time_limits)

            Glip_loss = 0.0
            if myargs.G_lip:
                value_loss, action_loss, dist_entropy, Glip_loss = agent.update(rollouts)
            else:
                value_loss, action_loss, dist_entropy = agent.update(rollouts)


                
            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % myargs.save_interval == 0
                    or j == num_updates - 1) and myargs.save_dir != "":
                save_path = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, f"actor_critic.pt"))

                if myargs.gail:
                    torch.save([
                        discr,
                        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                    ], os.path.join(save_path, f"discriminator.pt"))


            if j % myargs.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * myargs.num_processes * myargs.num_steps
                end = time.time()
                # print(
                #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                #     .format(j, total_num_steps,
                #             int(total_num_steps / (end - start)),
                #             len(episode_rewards), np.mean(episode_rewards),
                #             np.median(episode_rewards), np.min(episode_rewards),
                #             np.max(episode_rewards), dist_entropy, value_loss,
                #             action_loss))
            
            assert acc_expert_avg <= 1 
            assert acc_expert_pert_avg <= 1
            assert acc_policy_avg <= 1
            assert acc_policy_pert_avg <= 1


            with open(self.log_dir + "/" + self.log_file_name  + ".txt", "a") as file: 
                file.write(f'{j:>5} {total_num_steps:>8} {int(total_num_steps / (end - start)):>7} {len(episode_rewards):>4}\
                 {dist_entropy:.10} {value_loss:.10} {action_loss:.10} {np.mean(episode_rewards):.10} \
                 {np.median(episode_rewards):.10} {np.min(episode_rewards):.10}\
                 {np.max(episode_rewards):.10} {Glip_loss:.10}\
                 {disc_loss_lip_avg:.10} {disc_loss_expert_avg:.10} {disc_loss_policy_avg:.10} {disc_loss_total_avg:.10}\
                 {acc_expert_avg:.10} {acc_expert_pert_avg:.10} {acc_policy_avg:.10} {acc_policy_pert_avg:.10}\n')



            if (myargs.eval_interval is not None and len(episode_rewards) > 1
                    and j % myargs.eval_interval == 0 and j > 0):
                ob_rms = utils.get_vec_normalize(envs).ob_rms

                eval_episode_rewards_list = []
                num_processes = 1

                _ = evaluate(actor_critic, ob_rms, myargs.env_name, myargs.seed,
                     num_processes, self.eval_log_dir, self.log_file_name, self.device, myargs.gamma, myargs.adv_eval, None, myargs.lip_norm)

                if myargs.adv_eval:
                    pass
                    # for eval_noise in self.list_noise_levels:
                    #     eval_episode_rewards = evaluate(actor_critic, ob_rms, myargs.env_name, myargs.seed,
                    #         num_processes , self.eval_log_dir, "do-not-write-output", self.device, myargs.gamma, myargs.adv_eval, eval_noise)
                    #     eval_episode_rewards_list.append(eval_episode_rewards)
                    # eval_episode_rewards_list_all.append(eval_episode_rewards_list)   

        print(f"evaluate_policy_noisy_env() called ..........................")
        evaluate_policy_noisy_env_obj = evaluate_policy_noisy_env()
        evaluate_policy_noisy_env_obj.eval_func(self.myargs.save_name)

            # save_dict = {"noise_levels": self.list_noise_levels, "eval_episode_rewards_list_all": eval_episode_rewards_list_all}
            # with open(self.eval_log_dir +"/adv_eval_dict.pkl", "wb") as f:
            #     pickle.dump(save_dict, f)

            # print(psutil.virtual_memory()) 

            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

            # snapshot = tracemalloc.take_snapshot()
            # self.display_top(snapshot)

        # tracemalloc.stop()


class train_GT_policy():

    def __init__(self, kwargs, myargs):
        # ADDED BY FARZAN MEMARIAN 

        # ***************************** START
        # log_dir = os.path.expanduser(myargs.log_dir) + "/" + myargs.save_name 
        self.kwargs, self.myargs = kwargs, myargs
        self.init_params = []
        self.log_dir = myargs.log_dir + "/" + myargs.save_name 
        self.eval_log_dir = self.log_dir + "_eval"
        log_file_name = myargs.env_name
        # utils.cleanup_log_dir(log_dir)
        # utils.cleanup_log_dir(eval_log_dir)
        utils.create_dir(self.log_dir)
        utils.create_dir(self.eval_log_dir)

        # save_path_policy is for storing the trained model
        save_path_policy = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name)
        utils.create_dir(save_path_policy)

        # ***************************** END

        self._save_attributes_and_hyperparameters()


        torch.manual_seed(myargs.seed)
        torch.cuda.manual_seed_all(myargs.seed)
        np.random.seed(myargs.seed)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        device = myutils.assign_gpu_device(myargs)

        envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
            myargs.gamma, self.log_dir, device, allow_early_resets=False,  num_frame_stack=2,  **kwargs)

        # envs = gym.make(myargs.env_name, **kwargs)
        # envs = VecPyTorch(envs, device)
        # The followint block added by Farzan Memarian
        # envs = ProcgenEnv(num_envs=myargs.num_processes, env_name="heistpp", **kwargs)
        # if len(envs.observation_space.shape) == 3:
        #     envs = WarpFrame(envs, dict_space_key="rgb")
        # if len(envs.observation_space.shape) == 1 and myargs.do_normalize == "True":
        #     if gamma is None:
        #         envs = VecNormalize(envs, ret=False)
        #     else:
        #         envs = VecNormalize(envs, gamma=gamma)
        # envs = VecPyTorch(envs, device, myargs)

        if self.myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1", "Thrower-v2"]:
            hidden_size_policy = 10
        else:
            hidden_size_policy = 64

        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            device,
            base_kwargs={'recurrent': myargs.recurrent_policy, 'hidden_size': hidden_size_policy})
        actor_critic.to(device)

        if myargs.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                alpha=myargs.alpha,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                myargs.clip_param,
                myargs.ppo_epoch,
                myargs.num_mini_batch,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, myargs.value_loss_coef, myargs.entropy_coef, acktr=True)

        if myargs.gail:
            assert len(envs.observation_space.shape) == 1
            discr = gail.Discriminator(
                envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
                device)
            file_name = os.path.join(
                myargs.gail_experts_dir, "trajs_{}.pt".format(
                    myargs.env_name.split('-')[0].lower()))
            
            expert_dataset = gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20)
            drop_last = len(expert_dataset) > myargs.gail_batch_size
            gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=myargs.gail_batch_size,
                shuffle=True,
                drop_last=drop_last)

        rollouts = RolloutStorage(myargs.num_steps, myargs.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rews_from_info = deque(maxlen=10)

        start = time.time()
        num_updates = int(myargs.num_env_steps_tr) // myargs.num_steps // myargs.num_processes

        with open(self.log_dir + "/" + log_file_name  + ".txt", "w") as file: 
            file.write("Updates , num timesteps , FPS, number of Last training episodes, dist_entropy, value_loss, action_loss, mean reward, median reward, min reward, max reward \n")

        with open(self.eval_log_dir  + "/" + log_file_name  + "_eval.txt", "w") as file: 
            file.write("num_episodes, median_reward, max_reward \n")


        # UPDATE POLICY *****************************************************
        for j in range(num_updates):
            if j % 5 == 0 and j != 0:
                print (f'update number {j}, ...... {myargs.env_name}, total_time: {time.time()-start}')
            if myargs.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)


            for step in range(myargs.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                # Obser reward and next obs

                if myargs.env_name in ["CartPole-v0", "MountainCar-v0", "Acrobot-v1"]: # Added by Farzan Memarian
                    # action = action[0]
                    action_fed = torch.squeeze(action)
                else:
                    action_fed = action

                obs, reward, done, infos = envs.step(action_fed)

                # reward = torch.zeros((8,1))

                # envs.render()
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rews_from_info.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(copy.deepcopy(obs), recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            # Update value function at the end of episode
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            if myargs.gail:
                if j >= 10:
                    envs.venv.eval()

                gail_epoch = myargs.gail_epoch
                if j < 10:
                    gail_epoch = 100  # Warm up
                for _ in range(gail_epoch):
                    discr.update(gail_train_loader, rollouts,
                                 utils.get_vec_normalize(envs)._obfilt)

                for step in range(myargs.num_steps):
                    rollouts.rewards[step] = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], myargs.gamma,
                        rollouts.masks[step])

            rollouts.compute_returns(next_value, myargs.use_gae, myargs.gamma,
                                     myargs.gae_lambda, myargs.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save policy 
            if (j % myargs.save_interval == 0 or j == num_updates - 1) and myargs.save_dir != "":

                model_save_address = os.path.join(save_path_policy, myargs.save_name + "_" + str(j) + ".pt")
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], model_save_address)

            if j % myargs.log_interval == 0 and len(episode_rews_from_info) > 1:
                total_num_steps = (j + 1) * myargs.num_processes * myargs.num_steps
                end = time.time()
                # print(
                #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                #     .format(j, total_num_steps,
                #             int(total_num_steps / (end - start)),
                #             len(episode_rews_from_info), np.mean(episode_rews_from_info),
                #             np.median(episode_rews_from_info), np.min(episode_rews_from_info),
                #             np.max(episode_rews_from_info), dist_entropy, value_loss,
                #             action_loss))
                # with open(log_dir + "/" + log_file_name  + ".txt", "a") as file2: 
                #     file2.write("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                #     .format(j, total_num_steps,
                #             int(total_num_steps / (end - start)),
                #             len(episode_rews_from_info), np.mean(episode_rews_from_info),
                #             np.median(episode_rews_from_info), np.min(episode_rews_from_info),
                #             np.max(episode_rews_from_info), dist_entropy, value_loss,
                #             action_loss))

                with open(self.log_dir + "/" + log_file_name  + ".txt", "a") as file: 
                    file.write(f'{j:>5} {total_num_steps:>8} {int(total_num_steps / (end - start)):>7} {len(episode_rews_from_info):>4}\
                     {dist_entropy:.10} {value_loss:.10} {action_loss:.10} {np.mean(episode_rews_from_info):.10} \
                     {np.median(episode_rews_from_info):.10} {np.min(episode_rews_from_info):.10}\
                      {np.max(episode_rews_from_info):.10}  \n')

            # if (myargs.eval_interval is not None and len(episode_rewards) > 1
            #         and j % myargs.eval_interval == 0):
            #     ob_rms = utils.get_vec_normalize(envs).ob_rms
            #     evaluate(actor_critic, ob_rms, myargs.env_name, myargs.seed,
            #              myargs.num_processes, self.eval_log_dir, device)

    def _save_attributes_and_hyperparameters(self):
        print ("saving attributes_and_hyperparameters .....")



        with open(self.log_dir + "/init_params.txt", "w") as file:
            for key in self.init_params:
                file.write(f"{key} : {self.init_params[key]} \n" )

        args_dict = vars(self.myargs)
        with open(self.log_dir +"/args_dict.pkl", "wb") as f:
            pickle.dump(args_dict, f)
        with open(self.log_dir + "/myargs.txt", "w") as file:
            for key in args_dict:
                file.write(f"{key} : {args_dict[key]} \n" )
    
        with open(self.log_dir +"/kwargs.pkl", "wb") as f:
            pickle.dump(self.kwargs, f)    
        with open(self.log_dir + "/kwargs.txt", "w") as file:
            for key in self.kwargs:
                file.write(f"{key} : {self.kwargs[key]} \n" )


class evaluate_policy_noisy_env():

    def __init__(self):
        pass

    def eval_func(self, filename):
        myargs_address = f"./logs/{filename}/args_dict.pkl"
        with open(myargs_address, "rb") as f:
            myargs = pickle.load(f)

        from argparse import Namespace
        myargs = Namespace(**myargs)

        torch.manual_seed(myargs.seed+10)
        torch.cuda.manual_seed_all(myargs.seed+10)
        np.random.seed(myargs.seed+10)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        device = myutils.assign_gpu_device(myargs)

        # eval_log_dir = myargs.log_dir + "/" + myargs.save_name + "_eval" # this directory should already exist
        eval_log_dir = myargs.log_dir + "/" + myargs.save_name + "_eval" # this directory should already exist

        if myargs.lip_norm == "L_inf":
            list_noise_levels = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 3.0]
        elif myargs.lip_norm == "L_2":
            list_noise_levels = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]


        load_dir_policy = os.path.join("./trained_models", myargs.algo, f"{myargs.save_name}")
        names = listdir(load_dir_policy)

        # name = names[-1]
        address_load_policy = os.path.join(load_dir_policy,"actor_critic.pt")
        actor_critic, ob_rms = torch.load(address_load_policy, map_location=device)
        eval_episode_rewards_list = []

        for eval_noise in list_noise_levels:
            print(f"Call the evaluate function for noise_level:  {eval_noise}")
            num_processes = 1
            eval_episode_rewards = evaluate(actor_critic, ob_rms, myargs.env_name, myargs.seed, num_processes, 
                eval_log_dir, "do-not-write-output", device, myargs.gamma, adv_eval=True, eval_noise=eval_noise, norm=myargs.lip_norm)
            eval_episode_rewards_list.append(eval_episode_rewards)

        adv_eval_dict = {"noise_levels": list_noise_levels, "eval_episode_rewards_list": eval_episode_rewards_list}
        with open(eval_log_dir +"/adv_eval_dict.pkl", "wb") as f:
            pickle.dump(adv_eval_dict, f)


class train_Lipschitz_rew_from_ranks():

    def __init__(self, kwargs, myargs, init_params):
        if myargs.seed == -12345: # this means the seed is not provided by the user in command line arguments, it must be read from enviornment variables
            if os.getenv('SEED'):
                myargs.seed = int(os.getenv('SEED'))
            else:
                raise ValueError('SEED not provided as command line argument or as an enviornment variable')
        else:
            # seed is provided as command line argument and nothing needs to be done
            pass

        myargs.save_name = myargs.save_name + "-s" + str(myargs.pert_radius)


        self.kwargs = kwargs
        self.myargs = myargs
        self.init_params = init_params

        self.device = myutils.assign_gpu_device(myargs)

        # Read ranked trajs
        trajs_address = os.path.join("./ranked_trajs", myargs.load_name)
        self.save_path_new_trajs =  trajs_address + "/train" # overwrite parent class
        self.save_path_new_trajs =  trajs_address + "/val" # overwrite parent class

        with open(self.save_path_new_trajs+"/new_trajs_returns_list.pkl", "rb") as f:
            self.new_trajs_returns_list = pickle.load(f) # overwrite parent class
        with open(self.save_path_new_trajs_val+"/new_trajs_returns_list.pkl", "rb") as f:
            self.new_trajs_returns_list_val = pickle.load(f) # overwrite parent class

        self.log_dir = myargs.log_dir + "/" + myargs.save_name 
        eval_log_dir = self.log_dir + "_eval"
        self.log_file_name = myargs.env_name
        # utils.cleanup_log_dir(log_dir)
        # utils.cleanup_log_dir(eval_log_dir)
        utils.create_dir(self.log_dir)
        # utils.create_dir(eval_log_dir)

        # self.save_path_trained_models is for storing the trained model
        self.save_path_trained_models = os.path.join(myargs.save_dir, myargs.algo, myargs.save_name)
        if myargs.pretrain in ["yes","no"]:
            utils.create_dir(self.save_path_trained_models)

        # # Create forlder for tensorboard
        # self.writer = SummaryWriter(f'runs/visualization')

        torch.manual_seed(myargs.seed)
        torch.cuda.manual_seed_all(myargs.seed)
        np.random.seed(myargs.seed)

        if myargs.cuda and torch.cuda.is_available() and myargs.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)

        self.envs = make_vec_envs(myargs.env_name, myargs.seed, myargs.num_processes,
            myargs.gamma, self.log_dir, self.device, allow_early_resets=False, num_frame_stack=2, **kwargs)
        # envs = ProcgenEnv(num_envs=myargs.env_name, env_name="heistpp", **kwargs)

        # envs = gym.make(myargs.env_name, **kwargs)
        if self.myargs.env_name in ["MountainCar-v0", "Reacher-v2", "Acrobot-v1"]:
            hidden_size_policy = 10
        else:
            hidden_size_policy = 64

        self.actor_critic = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            self.device,
            base_kwargs={'recurrent': myargs.recurrent_policy, 'hidden_size': hidden_size_policy})
        self.actor_critic.to(self.device)


        if myargs.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                alpha=myargs.alpha,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'ppo':
            self.agent = algo.PPO(
                self.actor_critic,
                myargs.clip_param,
                myargs.ppo_epoch,
                myargs.num_mini_batch,
                myargs.value_loss_coef,
                myargs.entropy_coef,
                lr=myargs.lr,
                eps=myargs.eps,
                max_grad_norm=myargs.max_grad_norm)
        elif myargs.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, myargs.value_loss_coef, myargs.entropy_coef, acktr=True)

        # Initialize the reward function
        # self.reward_obj = reward_cl(myargs.num_processes, self.device, self.envs.observation_space.shape, myargs.rew_lr, myargs.rew_sign, myargs.rew_mag)
        self.num_rew_nets = myargs.num_rew_nets

        self.reward_objs = [reward_cl(self.device, self.envs.observation_space.shape, myargs.rew_lr, myargs.rew_sign, myargs.rew_mag) for i in range(self.num_rew_nets)]

        self.rollouts = RolloutStorage(myargs.num_steps, myargs.num_processes,
                              self.envs.observation_space.shape, self.envs.action_space,
                              self.actor_critic.recurrent_hidden_state_size)

        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        if myargs.pretrain in ["load", "no"]:
            list_appendix = ["train"]
        elif myargs.pretrain == "yes":
            list_appendix = ["pretrain", "train"]

        for appendix in list_appendix:
            with open(self.log_dir + f"/policy_stats.txt", "w") as file: 
                file.write("overal_tr_iter_idx, updates , num timesteps , FPS, number of Last training episodes, dist_entropy, value_loss, action_loss, mean reward, median reward, min reward, max reward \n")

            with open(self.log_dir + f"/rew_weights_stats.txt", "w") as file: 
                file.write("reward_mean, reward_std \n")

            with open(self.log_dir + f"/rew_losses.txt", "w") as file: 
                file.write("g_value \n")

            with open(self.log_dir + f"/rew_losses_val.txt", "w") as file: 
                file.write("g_value \n")

        with open(self.log_dir + "/buffer_stats.txt", "w") as file: 
            file.write("mean, range, std, mean_new, range_new, std_new \n")


        # print(f"Loading training trajectories ..... {self.myargs.save_name}")
        # with open(trajs_address + '/all_trajectories', 'rb') as f:
        #     # trajs = pickle.load(f)
        #     trajs_init = torch.load(f, map_location=self.device)

        # trajs_total_num = len(trajs_init)
        # traj_idxs_tr = np.arange(0, trajs_total_num, init_params['training_trajectory_skip'])
        # traj_idxs_val = traj_idxs_tr[:-3] + 1
        # demos_train_init = [trajs_init[idx] for idx in traj_idxs_tr]
        # demos_val_init = [trajs_init[idx] for idx in traj_idxs_val]

        # self.demos_train, self.demos_train_returns = myutils.trajs_calc_return_no_device(demos_train_init)
        # self.demos_val, self.demos_val_returns = myutils.trajs_calc_return_no_device(demos_val_init)

        # self.ranked_trajs_list, self.returns = myutils.cut_trajs_calc_return_no_device(trajs, self.init_params['demo_horizon'])

        #  FOLLOWING 4 LINES ARE USED IF WE USE OLD METHOD OF SAMPLLING SUM-TRAJECTORIES FROM INITIAL DEMONSTRATIONS
        # self.demos_subsampled_list, self.demos_returns_subsampled = myutils.subsample_demos(trajs, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])
        # self.demos_subsampled_list_val, self.demos_returns_subsampled_val = myutils.subsample_demos(trajs_val, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])

        # self.demos_subsampled_list, self.demos_returns_subsampled = myutils.subsample_demos_true_return(trajs, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])
        # self.demos_subsampled_list_val, self.demos_returns_subsampled_val = myutils.subsample_demos_true_return(trajs_val, self.init_params['subsample_length'], self.init_params['subsample_increment'], self.init_params['demo_horizon'])

        self.deque_S = deque(maxlen = self.init_params['deque_S_size']) # deque for storing new trajs produced by the policy
        # self.deque_S = deque()
        # self._populate_deque_S()

        # self.new_trajs_returns_list = deque(maxlen= self.init_params['size_of_new_trajs_list'])
        self._save_attributes_and_hyperparameters()

    def _populate_deque_S(self):
        for idx in range(len(self.new_trajs_returns_list)):
            traj = torch.load(self.save_path_new_trajs+f"/traj_{idx}.pt", map_location=self.device)
            states = [item[0] for item in traj]
            self.deque_S.extend(states)

    def train(self):  

        # sample trajectories from the new policy and add to the buffer
        # produced_trajs = myutils.produce_trajs_from_policy_sparsified_reward(self.actor_critic, self.myargs.sparseness, self.init_params['num_trajs_produced_each_iter'], self.init_params['produced_traj_length'], self.kwargs, self.myargs)
        # produced_trajs = self.produce_trajs_from_policy(self.actor_critic, self.init_params['num_trajs_produced_each_iter'], self.init_params['produced_traj_length'], self.kwargs, self.myargs)
        # new_states = myutils.extract_states_from_trajs(produced_trajs)
        # self.deque_S.extend(new_states)

        for overal_tr_iter_idx in range(self.init_params["num_overal_updates"]):
            # TO DO: Strike a balance between using initial set of ranked demos, and the new set
            print(f"overal_tr_iter_idx: {overal_tr_iter_idx}")
            # # Update g according to demos
            # ranked_traj_list, traj_returns = self.demos_train, self.demos_train_returns

            pairs, returns, pair_select_time_total, rew_update_time_total = self.grad_g_theta_update(overal_tr_iter_idx, self.init_params['num_demos_batches_train'], self.num_rew_nets, 
                                self.init_params['batch_size'], demos_or_policy='demos', pretrain_or_train="train", discounted_rew=self.myargs.discounted_rew)

            # populate the self.deque_S by states that are recently used for updating the reward
            self.deque_S = deque(maxlen = self.init_params['deque_S_size']) # deque for storing new trajs produced by the policy
            for pair in pairs:
                for traj in pair:
                    # states = [item for item in traj]
                    self.deque_S.extend(traj[:])


            # Update g according to states
            chosen_state_idxs = np.random.choice(len(self.deque_S), size=self.myargs.num_states_update, replace=False)
            chosen_states = [self.deque_S[idx] for idx in chosen_state_idxs]
            training_states_and_perturbed_list = []

            for state in chosen_states:
                state = state.to(device=self.device)
                if len(state.size()) == 1:
                    noise_dim = state.size()[0]
                else:
                    noise_dim = state.size()[1]
                perturbed_states = [state + torch.from_numpy(np.random.normal(0,1,noise_dim)).to(device=self.device).type(torch.float32).reshape(1,noise_dim) for _ in range(self.myargs.num_perturbations)]
                # perturbed_states = [state[0,:] + torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0])) for _ in range(self.myargs.num_perturbations)]
                training_states_and_perturbed_list.append((state, perturbed_states))

            self.grad_second_term_theta_update(training_states_and_perturbed_list, self.myargs.reg_beta)

            # UPDATE POLICY *****************************************************
            # num_pol_updates_tr = int(self.myargs.num_env_steps_tr) // self.myargs.num_steps // self.myargs.num_processes
            # self.update_policy(overal_tr_iter_idx, num_pol_updates_tr, pretrain_or_train="train")


            # save reward after each self.init_params['save_reward_int'] overal iterations
            if overal_tr_iter_idx % self.init_params['save_reward_int'] == 0 and overal_tr_iter_idx != 0:
                for reward_idx, reward_obj in enumerate(self.reward_objs):
                    model_save_address = os.path.join(self.save_path_trained_models, self.myargs.save_name + f"_reward_{reward_idx}.pt")
                    torch.save({'model_state_dict': reward_obj.reward_net.state_dict(),
                                'optimizer_state_dict': reward_obj.optimizer.state_dict()}, 
                                model_save_address)

                # model_save_address = os.path.join(self.save_path_trained_models, self.myargs.save_name + f"_policy_iter_{overal_tr_iter_idx}.pt")
                # torch.save([
                #     self.actor_critic,
                #     getattr(utils.get_vec_normalize(self.envs), 'ob_rms', None)
                # ], model_save_address) 

            # # sample trajectories from the new policy and add to the buffer
            # produced_trajs = self.produce_trajs_from_policy(self.actor_critic, self.init_params['num_trajs_produced_each_iter'], self.init_params['produced_traj_length'], self.kwargs, self.myargs)
            # new_states = myutils.extract_states_from_trajs(produced_trajs)
            # self.deque_S.extend(new_states)

    def grad_second_term_theta_update(self, training_states_and_perturbed_list, reg_beta):
        criterion = torch.nn.L1Loss()
        
        for rew_obj_idx, reward_obj in enumerate(self.reward_objs):

            X = []
            Y = []
            max_idxs = []
            # find max_idxs for perturbations
            for state, perturbed_states in training_states_and_perturbed_list:
                with torch.no_grad():
                    target = reward_obj.reward_net(state)
                    max_idx = np.argmax([abs(target.item() - reward_obj.reward_net(perturbed_state).item()) for perturbed_state in perturbed_states])
                    Y.append(target)
                    max_idxs.append(max_idx)

            training_states_list = [item[0] for item in training_states_and_perturbed_list]
            reward_obj.reward_net.zero_grad()

            for state_idx, max_idx in enumerate(max_idxs):
                X.append(reward_obj.reward_net(training_states_and_perturbed_list[state_idx][1][max_idx][0]))

            X_tensor = torch.cat(X, dim=0)
            Y_tensor = torch.cat(Y, dim=0)
            loss = reg_beta*criterion(X_tensor, Y_tensor)     
            loss.backward()
            reward_obj.optimizer.step()

    def update_policy(self, overal_tr_iter_idx, num_updates, pretrain_or_train):
        kwargs, myargs = self.kwargs, self.myargs
        episode_rews_from_info = deque(maxlen=myargs.num_processes)
        episode_rew_net_return = deque(maxlen=myargs.num_processes)
        start = time.time()

        if myargs.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                self.agent.optimizer, overal_tr_iter_idx, self.init_params["num_overal_updates"],
                self.agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)
        elif myargs.use_increase_decrease_lr_pol:
            utils.update_linear_schedule_increase_decrease(
                self.agent.optimizer, overal_tr_iter_idx, self.init_params["num_overal_updates"],
                self.agent.optimizer.lr if myargs.algo == "acktr" else myargs.lr)            

        unrolled_trajs_all = []
        no_run, no_run_no_cntr = self._specify_env_rew_type()

        for j in range(num_updates):

            # at each j, each of the policies will be unrolled for up to myargs.num_steps steps, or until they get to an
            # absorbing state (like failure)

            return_nets_episode = torch.zeros((myargs.num_rew_nets, myargs.num_processes), device=self.device)
            return_GT_episode_cntr = np.zeros(myargs.num_processes)
            return_GT_episode_run = np.zeros(myargs.num_processes) 
            num_succ_run_forward = np.zeros(myargs.num_processes) 
            displacement_forward_till_rew = np.zeros(myargs.num_processes) 
            steps_till_rew = np.zeros(myargs.num_processes) 
            displacement_forward_episode_total = np.zeros(myargs.num_processes) 
            num_succ_not_done = np.zeros(myargs.num_processes)
            return_sparse_episode = np.zeros(myargs.num_processes)
            return_dense_plus_cntr_episode = np.zeros(myargs.num_processes)
            num_succ_run_forward_avg_steps = np.zeros(myargs.num_processes)
            num_succ_not_done_avg_steps = np.zeros(myargs.num_processes)
            num_steps_taken_to_rew = np.zeros(myargs.num_processes)
            displacement_total_from_infos = np.zeros(myargs.num_processes)


            for step in range(myargs.num_steps):

                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                obs, reward_GT, done, infos = self.envs.step(action)


                with torch.no_grad():
                    rew_nets_step_list = [reward_obj.reward_net(obs) for reward_obj in self.reward_objs]

                for rew_idx in range(len(self.reward_objs)):
                    return_nets_episode[rew_idx, :] += rew_nets_step_list[rew_idx].reshape(myargs.num_processes)

                # add rewards of the networks to the network calculated returns

 

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rews_from_info.append(info['episode']['r'])
                        # info stores the undiscounted return of each trajectory

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])

                rews_nets_step = torch.mean(torch.cat(rew_nets_step_list, dim=1), dim=1)
                
                if myargs.rew_cntr == "True":
                    total_rews_step = self.myargs.rew_coeff * rews_nets_step + self.myargs.cntr_coeff * torch.tensor(rews_cntr_step, device=self.device)
                    total_rews_GT_step = reward_sparse.to(self.device) + self.myargs.cntr_coeff * torch.tensor(rews_cntr_step, device=self.device)
                elif myargs.rew_cntr == "False":
                    total_rews_step = rews_nets_step
                    total_rews_GT_step = reward_sparse
                total_rews_step_torch = torch.unsqueeze(total_rews_step, dim=1)

                for idx_proc, _done in enumerate(done):
                    if not _done:
                        return_GT_episode_cntr[idx_proc] +=  rews_cntr_step[idx_proc] # * myargs.gamma**step 
                        return_GT_episode_run[idx_proc]  +=  rews_run_step[idx_proc] # * myargs.gamma**step 

                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, total_rews_step_torch, masks, bad_masks)


            return_GT_episode_total_my_calc = return_GT_episode_cntr + return_GT_episode_run

            #  END OF EIPISODE OR MAXIMUM NUMBER OF STEPS
            for idx in range(myargs.num_processes):
                episode_rew_net_return.append(torch.mean(return_nets_episode[:, idx]).item())

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value, myargs.use_gae, myargs.gamma,
                                 myargs.gae_lambda, myargs.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            self.rollouts.after_update()
            if len(episode_rews_from_info) > 1:
                total_num_steps = (j + 1) * myargs.num_processes * myargs.num_steps
                end = time.time()
                with open(self.log_dir + f"/policy_stats.txt", "a") as file: 
                    file.write( 
                        f'{overal_tr_iter_idx:>5} {j:>5} \
                        {total_num_steps:>8} {int(total_num_steps / (end - start)):.10f} \
                        {len(episode_rews_from_info):>4} {dist_entropy:.10} \
                        {value_loss:.10} {action_loss:.10}\
                        {np.mean(episode_rews_from_info):.10} {np.median(episode_rews_from_info):.10} \
                        {np.min(episode_rews_from_info):.10} {np.max(episode_rews_from_info):.10} {np.std(episode_rews_from_info):.10}\
                        {np.mean(episode_rew_net_return):.10} {np.std(episode_rew_net_return):.10} \
                        {np.mean(return_GT_episode_cntr):.10} {np.std(return_GT_episode_cntr):.10} \
                        {np.mean(return_GT_episode_run):.10}  {np.std(return_GT_episode_run):.10} \
                        {np.mean(return_dense_plus_cntr_episode):.10} {np.std(return_dense_plus_cntr_episode):.10} \
                        {np.mean(num_succ_not_done_avg_steps):.10} {np.std(num_succ_not_done_avg_steps):.10} \n' )


def main():
    myargs, kwargs = arguments.get_args()
    # print("After get_args")

    if myargs.main_function == "train_GT_policy":
        train_GT_policy(kwargs, myargs)

    if myargs.main_function == "train_GAIL":
            train_GAIL(kwargs, myargs)

    elif myargs.main_function == "produce_ranked_trajs":
        produce_ranked_trajs(kwargs, myargs)

    elif myargs.main_function == "produce_ranked_trajs_sparse":
        init_params = arguments.get_init(myargs)
        produce_ranked_trajs_sparse(kwargs, myargs, init_params)

    elif myargs.main_function == "train_sparse_rank":
        init_params = arguments.get_init(myargs)
        train_obj = train_sparse_rank(kwargs, myargs, init_params)
        train_obj.train()

    elif myargs.main_function == "train_baseline_sparse_rew":
        init_params = arguments.get_init_baseline(myargs)
        train_obj = train_baseline_sparse_rew(kwargs, myargs, init_params)
        train_obj.train()       

    elif myargs.main_function == "evaluate_policy_noisy_env":
        evaluate_policy_noisy_env_obj = evaluate_policy_noisy_env()
        evaluate_policy_noisy_env_obj.eval_func(myargs.eval_name)


    elif myargs.main_function == "visualize_policy":
        train_obj = visualize_policy(kwargs, myargs)


if __name__ == "__main__":
    main()






