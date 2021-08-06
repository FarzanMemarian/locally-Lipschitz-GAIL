import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
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

            acc_expert_pert_avg += 0
            acc_policy_pert_avg += 0


            gail_loss = expert_loss + policy_loss 
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            lip_loss_avg += 0
            expert_loss_avg += expert_loss.item()
            policy_loss_avg += policy_loss.item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return (loss/n, lip_loss_avg/n, expert_loss_avg/n, policy_loss_avg/n, 
            acc_expert_avg/n, acc_expert_pert_avg/n, acc_policy_avg/n, acc_policy_pert_avg/n)


    # def update(self, expert_loader, rollouts, obsfilt=None):
    #     self.train()

    #     policy_data_generator = rollouts.feed_forward_generator(
    #         None, mini_batch_size=expert_loader.batch_size)

    #     loss = 0
    #     expert_loss_avg = 0
    #     policy_loss_avg = 0
    #     n = 0
    #     for expert_batch, policy_batch in zip(expert_loader,
    #                                           policy_data_generator):
    #         policy_state, policy_action = policy_batch[0], policy_batch[2]
    #         policy_d = self.trunk(
    #             torch.cat([policy_state, policy_action], dim=1))

    #         expert_state, expert_action = expert_batch
    #         expert_state = obsfilt(expert_state.numpy(), update=False)
    #         expert_state = torch.FloatTensor(expert_state).to(self.device)
    #         expert_action = expert_action.to(self.device)
    #         expert_d = self.trunk(
    #             torch.cat([expert_state, expert_action], dim=1))
    #         acc_expert = self.calc_accuracy(expert_d, "expert")
    #         acc_expert_avg += acc_expert

    #         expert_loss = F.binary_cross_entropy_with_logits(
    #             expert_d,
    #             torch.ones(expert_d.size()).to(self.device))
    #         policy_loss = F.binary_cross_entropy_with_logits(
    #             policy_d,
    #             torch.zeros(policy_d.size()).to(self.device))

    #         gail_loss = expert_loss + policy_loss
    #         grad_pen = self.compute_grad_pen(expert_state, expert_action,
    #                                          policy_state, policy_action)

    #         loss += (gail_loss + grad_pen).item()
    #         expert_loss_avg += expert_loss.item()
    #         policy_loss_avg += policy_loss.item()
    #         n += 1

    #         self.optimizer.zero_grad()
    #         (gail_loss + grad_pen).backward()
    #         self.optimizer.step()
    #     return loss/n, 0, expert_loss_avg/n, policy_loss_avg/n

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


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, noisy_training, train_noise, norm, num_trajectories=4, subsample_frequency=20):
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

        if noisy_training in ["DG", "D"]:
            self.trajectories['states'] = self.add_noise_to_obs(self.trajectories['states'], train_noise, norm)
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]

    def add_noise_to_obs(self, obs, radius, norm):
        obs_size = obs.size()
        # delta has a norm ifinity smaller than radius
        if norm == "L_inf":
            delta = ((torch.rand(obs_size).to(obs.device) - 0.5 ) * radius / 0.5)
        elif norm == "L_2":
            means = torch.zeros(obs_size)
            stds = torch.ones(obs_size)
            delta = torch.normal(means, stds).to(obs.device)
            norms = torch.unsqueeze(torch.norm(delta, dim=2), dim=2)
            norms_cat = torch.cat([norms for _ in range(obs_size[-1])], dim=2)
            delta = delta * radius / norms_cat
        else:
            raise Exception("L_p norm other than L_infinity and L_2 not implemented yet")
        obs_pertubed = obs + delta 
        return obs_pertubed
