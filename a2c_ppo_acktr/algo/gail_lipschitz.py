import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
import torch.optim as optim

import copy

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, pert_radius, lip_coeff, D_lipschitz, lip_norm, lr_proj_ascent=1, num_steps_pert=10):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.lip_coeff = lip_coeff
        self.pert_radius = pert_radius
        self.lip_norm = lip_norm
        self.num_steps_pert = num_steps_pert
        if self.lip_norm  == "L_inf":
            self.lr_proj_ascent = lr_proj_ascent
        elif self.lip_norm  == "L_2":
            self.lr_proj_ascent = self.pert_radius * 2 / self.num_steps_pert
            # self.lr_proj_ascent = lr_proj_ascent
        self.D_lipschitz = D_lipschitz



    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True 

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()
        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        lip_loss_avg = 0
        expert_loss_avg = 0
        policy_loss_avg = 0
        acc_expert_avg = 0
        acc_expert_pert_avg = 0
        acc_policy_avg = 0
        acc_policy_pert_avg = 0

        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))
            acc_policy = self.calc_accuracy(policy_d, "policy")
            acc_policy_avg += acc_policy

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))
            acc_expert = self.calc_accuracy(expert_d, "expert")
            acc_expert_avg += acc_expert

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            # Calculating the lipschitz loss
            expert_state_perturbed = self.perform_projected_grad_ascent(expert_state, expert_action, self.lip_norm)
            policy_state_perturbed = self.perform_projected_grad_ascent(policy_state, policy_action, self.lip_norm)

            pert_logits_expert = self.trunk(torch.cat([expert_state_perturbed, expert_action], dim=1))
            pert_logits_policy = self.trunk(torch.cat([policy_state_perturbed, policy_action], dim=1))

            acc_expert_pert = self.calc_accuracy(pert_logits_expert, "expert")
            acc_expert_pert_avg += acc_expert_pert

            acc_policy_pert = self.calc_accuracy(pert_logits_policy, "policy")
            acc_policy_pert_avg += acc_policy_pert

            loss_lip_expert = torch.norm(pert_logits_expert-expert_d, p=1)
            loss_lip_policy = torch.norm(pert_logits_policy-policy_d, p=1)

            lip_loss = (loss_lip_expert + loss_lip_policy)/2
            if self.D_lipschitz:
                gail_loss = expert_loss + policy_loss + self.lip_coeff * lip_loss
            else:
                gail_loss = expert_loss + policy_loss 

            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            lip_loss_avg += lip_loss.item()
            expert_loss_avg += expert_loss.item()
            policy_loss_avg += policy_loss.item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return (loss/n, lip_loss_avg/n, expert_loss_avg/n, policy_loss_avg/n, 
            acc_expert_avg/n, acc_expert_pert_avg/n, acc_policy_avg/n, acc_policy_pert_avg/n)

    def predict_reward_original(self, state, action, gamma, masks, update_rms=True):
        """
        This is the original predict_reward function that Farzan has renamed by adding _original
        Instead of this, I have added a new function predict_reward
        """
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        """
        This is a function added by Farzan Memarian, instead of predict_reward_original
        """
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward 

    def calc_accuracy(self, logits, source):
        s = torch.sigmoid(logits)
        total_num = s.size()[0]
        
        if source == "expert":
            bools = s >= 0.5
        elif source == "policy":
            bools = s < 0.5

        accuracy = torch.sum(bools).item() / float(total_num)
        return accuracy




    def perform_projected_grad_ascent(self, obs_batch, action_batch, lip_norm): 
        # Added by Farzan: calculate lipschitz loss
        # *****************************************

        # self.actor_critic_lip.load_state_dict(copy.deepcopy(self.actor_critic.state_dict()))
        trunk_lip = copy.deepcopy(self.trunk)

        # Bellow we make sure parameters of trunk_lip are frozen in this stage where we 
        # want to find the perturbation with maximum change in the output distribution
        for p in trunk_lip.parameters(): 
            p.requires_grad = False
        trunk_lip.eval()

            # In the above line we copy the self.actor_critic into self.trunk_lip
            # so that the operations done for lipschitz regularization does not change 
            # the gradients with respect to the weights of self.actor_critic
        obs_batch_lip = obs_batch.clone().detach()
        obs_size = obs_batch_lip.size()



        # this is delta_0 to begin projected gradient ascent, 
        # it is by definition in the feasibility set defined by the epsilon L_infinity or L_2 ball
        if lip_norm == "L_inf":
            delta_lip = ((torch.rand(obs_size) - 0.5 ) * self.pert_radius / 0.5).to(obs_batch.device)
        elif lip_norm == "L_2":
            delta_lip = torch.renorm(torch.rand(obs_size).to(obs_batch.device), p=2, dim=0, maxnorm=self.pert_radius)

        # obs = torch.unsqueeze(obs,dim=0)
        base_logits = trunk_lip(torch.cat([obs_batch_lip, action_batch], dim=1))

        for iter in range(10):
            delta_lip.requires_grad_()
            optimizer_lip = optim.SGD([delta_lip], lr=self.lr_proj_ascent)
            pert_logits = trunk_lip(torch.cat([obs_batch_lip+delta_lip, action_batch], dim=1))
            loss_lip = -torch.norm(pert_logits-base_logits, p=1) # The minus is because we want to perform gradient ascent not descent
            optimizer_lip.zero_grad()
            loss_lip.backward(retain_graph=True)
            optimizer_lip.step()

            with torch.no_grad():
                if lip_norm == "L_inf":
                    delta_lip = torch.clamp(delta_lip, -self.pert_radius, self.pert_radius) # projecting to the L_infinity ball
                elif lip_norm == "L_2":
                    delta_lip = torch.renorm(delta_lip, p=2, dim=0, maxnorm=self.pert_radius) # projecting to the L_2 ball

        obs_batch_lip_perturbed = obs_batch_lip+delta_lip.detach()

        return obs_batch_lip_perturbed
        # *****************************************



class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]
