import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from torch.distributions.kl import kl_divergence



# def add_noise_to_obs(obs, radius, norm):
#     obs_size = obs.size()
    
#     if norm == "L_inf":
#         # delta has a norm ifinity smaller than radius
#         delta = ((torch.rand(obs_size).to(obs.device) - 0.5 ) * radius / 0.5)
#     elif norm == "L_2":
#         delta = torch.rand(obs_size).to(obs.device)
#         delta = delta * radius / torch.norm(delta, p=2) # this makes sure that the L_2 norm is exacly equal to radius
#     else:
#         raise Exception("L_p norm other than L_infinity and L_2 not implemented yet")
        
#     obs_pertubed = obs + delta 
#     return obs_pertubed

def add_noise_to_obs(obs, radius, norm):
    obs_size = obs.size()
    
    if norm == "L_inf": # delta has a norm ifinity smaller than radius
        delta = ((torch.rand(obs_size).to(obs.device) - 0.5 ) * radius / 0.5)
    elif norm == "L_2": # delta has a norm ifinity exactly equal to the radius
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


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir, log_file_name,
             device, gamma, adv_eval, eval_noise, norm):

    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              gamma, eval_log_dir, device, allow_early_resets=False,  num_frame_stack=None)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    if adv_eval:
        obs = add_noise_to_obs(obs, eval_noise, norm)

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)



    lip_measure = 0
    obs_lip_batch_size = 128
    obs_batch_list = []
    obs_perturbed_batch_list = []
    Jeffery_div_loss_list = []
    while len(eval_episode_rewards) < 30:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs_original, _, done, infos = eval_envs.step(action)
        if adv_eval:
            obs = add_noise_to_obs(obs_original, eval_noise, norm)
            obs_batch_list.append(obs_original)
            obs_perturbed_batch_list.append(obs)
            if len(obs_batch_list) == obs_lip_batch_size:
                obs_batch = torch.cat(obs_batch_list, dim=0)
                obs_pertubed_batch = torch.cat(obs_perturbed_batch_list, dim=0)
                with torch.no_grad():
                    dist_base = actor_critic.get_distribution(obs_batch, 1, 1) # we set mask and recurrent parameters to 1, they don't matter here
                    dist_perturbed = actor_critic.get_distribution(obs_pertubed_batch, 1, 1) # we set mask and recurrent parameters to 1, they don't matter here
                    # M = 0.5 * (dist_perturbed + dist_base)
                    # JS_div_loss = torch.mean(kl_divergence(dist_perturbed, M) + kl_divergence(dist_base, M))/2 
                    Jeffery_div_loss = torch.mean(kl_divergence(dist_perturbed, dist_base) + kl_divergence(dist_base, dist_perturbed))/2 
                Jeffery_div_loss_list.append(Jeffery_div_loss.item())
                obs_batch_list = []
                obs_perturbed_batch_list = []
        else:
            obs = obs_original


        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    if log_file_name != "do-not-write-output":
        with open(eval_log_dir  + "/" + log_file_name  + ".txt", "a") as file:  
            file.write(f' {len(eval_episode_rewards):>4}  {np.median(eval_episode_rewards):.10} {np.max(eval_episode_rewards):.10} \n')

    if adv_eval:
        Jeffery_div_loss_normalized_mean = np.mean(Jeffery_div_loss_list)/eval_noise
    else:
        Jeffery_div_loss_normalized_mean = 0
    return [ np.median(eval_episode_rewards), np.max(eval_episode_rewards), Jeffery_div_loss_normalized_mean]


# def evaluate_discriminator(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir, log_file_name,
#              device, gamma, adv_eval, eval_noise):

#     eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
#                               gamma, eval_log_dir, device, allow_early_resets=False,  num_frame_stack=None)

#     vec_norm = utils.get_vec_normalize(eval_envs)
#     if vec_norm is not None:
#         vec_norm.eval()
#         vec_norm.ob_rms = ob_rms

#     eval_episode_rewards = []

#     obs = eval_envs.reset()
#     if adv_eval:
#         obs = add_noise_to_obs(obs, eval_noise, "infinity")

#     eval_recurrent_hidden_states = torch.zeros(
#         num_processes, actor_critic.recurrent_hidden_state_size, device=device)
#     eval_masks = torch.zeros(num_processes, 1, device=device)



#     lip_measure = 0
#     obs_lip_batch_size = 128
#     obs_batch_list = []
#     obs_perturbed_batch_list = []
#     Jeffery_div_loss_list = []
#     while len(eval_episode_rewards) < 30:
#         with torch.no_grad():
#             _, action, _, eval_recurrent_hidden_states = actor_critic.act(
#                 obs,
#                 eval_recurrent_hidden_states,
#                 eval_masks,
#                 deterministic=True)

#         # Obser reward and next obs
#         obs_original, _, done, infos = eval_envs.step(action)
#         if adv_eval:
#             obs = add_noise_to_obs(obs_original, eval_noise, "infinity")
#             obs_batch_list.append(obs_original)
#             obs_perturbed_batch_list.append(obs)
#             if len(obs_batch_list) == obs_lip_batch_size:
#                 obs_batch = torch.cat(obs_batch_list, dim=0)
#                 obs_pertubed_batch = torch.cat(obs_perturbed_batch_list, dim=0)
#                 with torch.no_grad():
#                     dist_base = actor_critic.get_distribution(obs_batch, 1, 1) # we set mask and recurrent parameters to 1, they don't matter here
#                     dist_perturbed = actor_critic.get_distribution(obs_pertubed_batch, 1, 1) # we set mask and recurrent parameters to 1, they don't matter here
#                     Jeffery_div_loss = torch.mean(kl_divergence(dist_perturbed, dist_base) + kl_divergence(dist_base, dist_perturbed))/2 
#                 Jeffery_div_loss_list.append(Jeffery_div_loss.item())
#                 obs_batch_list = []
#                 obs_perturbed_batch_list = []
#         else:
#             obs = obs_original


#         eval_masks = torch.tensor(
#             [[0.0] if done_ else [1.0] for done_ in done],
#             dtype=torch.float32,
#             device=device)

#         for info in infos:
#             if 'episode' in info.keys():
#                 eval_episode_rewards.append(info['episode']['r'])

#     eval_envs.close()

#     if log_file_name != "do-not-write-output":
#         with open(eval_log_dir  + "/" + log_file_name  + ".txt", "a") as file:  
#             file.write(f' {len(eval_episode_rewards):>4}  {np.median(eval_episode_rewards):.10} {np.max(eval_episode_rewards):.10} \n')

#     if adv_eval:
#         Jeffery_div_loss_normalized_mean = np.mean(Jeffery_div_loss_list)/eval_noise
#     else:
#         Jeffery_div_loss_normalized_mean = 0
#     return [ np.median(eval_episode_rewards), np.max(eval_episode_rewards), Jeffery_div_loss_normalized_mean]
