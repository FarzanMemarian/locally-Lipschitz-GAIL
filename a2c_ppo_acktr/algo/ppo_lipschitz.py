import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence
import time
import copy

import os
import psutil

class PPO_Lipschitz():
  
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lip_coeff,
                 pert_radius,
                 lip_norm,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 num_steps_pert=10,
                 lr_proj_ascent=1
                 ):

        self.actor_critic = actor_critic
        # self.actor_critic_lip = actor_critic_lip


        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lip_coeff = lip_coeff
        self.pert_radius = pert_radius
        self.lip_norm = lip_norm
        self.num_steps_pert = num_steps_pert
        if lip_norm == "L_inf":
            self.lr_proj_ascent = lr_proj_ascent
        elif lip_norm == "L_2":
            self.lr_proj_ascent = self.pert_radius * 2 / self.num_steps_pert
            # self.lr_proj_ascent = lr_proj_ascent

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def memory_usage_psutil(self):
        # return the memory usage in GB
        import psutil
        process = psutil.Process()
        mem = process.memory_info().rss / 1073741824
        return mem


    def update(self, rollouts):


        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        lip_loss_epoch = 0

        for e in range(self.ppo_epoch):
            # print(f"ppo_epoch: {e} *****************")
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                obs_batch_lip_perturbed  = self.perform_projected_grad_ascent(obs_batch, self.lip_norm)

                # now compute lip_loss as a function of policy weights, note that we want to minimize this loss 
                dist_base = self.actor_critic.get_distribution(obs_batch, 1, 1)
                dist_perturbed = self.actor_critic.get_distribution(obs_batch_lip_perturbed, 1, 1)
                lip_loss = torch.mean(kl_divergence(dist_perturbed, dist_base) + kl_divergence(dist_base, dist_perturbed))/2 
                # print(f"Memory usage in GB line 139:   {self.memory_usage_psutil()}")

                self.optimizer.zero_grad()
                non_lipchitz_loss = (value_loss * self.value_loss_coef + action_loss - 
                dist_entropy * self.entropy_coef)
                lip_regularized_loss = non_lipchitz_loss + self.lip_coeff * lip_loss
                lip_regularized_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()


                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                lip_loss_epoch += lip_loss.item()


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        lip_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, lip_loss_epoch

    def perform_projected_grad_ascent(self, obs_batch, lip_norm): 
        # Added by Farzan: calculate lipschitz loss
        # *****************************************

        # self.actor_critic_lip.load_state_dict(copy.deepcopy(self.actor_critic.state_dict()))
        actor_critic_lip = copy.deepcopy(self.actor_critic)

        # Bellow we make sure parameters of actor_critic_lip are frozen in this stage where we 
        # want to find the perturbation with maximum change in the output distribution
        for p in actor_critic_lip.parameters(): 
            p.requires_grad = False
        actor_critic_lip.eval()

            # In the above line we copy the self.actor_critic into self.actor_critic_lip
            # so that the operations done for lipschitz regularization does not change 
            # the gradients with respect to the weights of self.actor_critic
        obs_batch_lip = obs_batch.clone().detach()
        obs_size = obs_batch_lip.size()

        # this is delta_0 to begin projected gradient ascent, 
        # it is by definition in the feasibility set defined by the L_2 or L_infinity ball
        if lip_norm == "L_inf":
            delta_lip = ((torch.rand(obs_size) - 0.5 ) * self.pert_radius / 0.5).to(obs_batch.device)
        elif lip_norm == "L_2":
            delta_lip = torch.renorm(torch.rand(obs_size).to(obs_batch.device), p=2, dim=0, maxnorm=self.pert_radius)
              # in the above line, the renorm makes sure that the L_2 norm of delta_lip is smaller than self.pert_radius

        # obs = torch.unsqueeze(obs,dim=0)
        dist_base = actor_critic_lip.get_distribution(obs_batch_lip, 1, 1) # we set mask and recurrent parameters to 1, they don't matter here

        for iter in range(self.num_steps_pert):
            delta_lip.requires_grad_()
            optimizer_lip = optim.SGD([delta_lip], lr=self.lr_proj_ascent)
            dist = actor_critic_lip.get_distribution(obs_batch_lip+delta_lip, 1, 1)
            neg_Jeffery_div_loss = -torch.mean(kl_divergence(dist, dist_base) + kl_divergence(dist_base, dist))/2 
                            # the minus sign above makes it the loss for projected gradient ascent rather than descent

            # M = 0.5 * (dist + dist_base)
            # neg_JS_div_loss = -torch.mean(kl_divergence(dist, M) + kl_divergence(dist_base, M))/2 
            
            optimizer_lip.zero_grad()
            neg_Jeffery_div_loss.backward(retain_graph=True)
            optimizer_lip.step()

            with torch.no_grad():
                if lip_norm == "L_inf":
                    delta_lip = torch.clamp(delta_lip, -self.pert_radius, self.pert_radius) # projecting to the L_infinity ball
                elif lip_norm == "L_2":
                    delta_lip = torch.renorm(delta_lip, p=2, dim=0, maxnorm=self.pert_radius) # projecting to the L_2 ball
                

        obs_batch_lip_perturbed = obs_batch_lip+delta_lip.detach()

        return obs_batch_lip_perturbed
        # *****************************************

