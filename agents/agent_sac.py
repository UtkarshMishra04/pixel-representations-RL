import numpy as np
from numpy.core.numeric import tensordot
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import data_augs as rad
from agents.auxiliary_funcs import BaseSacAgent
from multistep_dynamics import MultiStepDynamicsModel
import multistep_utils as mutils

class PixelSacAgent(BaseSacAgent):
    """Learning Representations of Pixel Observations with SAC + Self-Supervised Techniques.."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        horizon,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_lr=1e-3,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs = '',
        use_metric_loss=False
    ):

        print('########################################################################')
        print('################### Starting Case 0: Baseline Agent ####################')
        print('###################### Reward: No; Transition: No ######################')
        print('########################################################################')
        
        super().__init__(
            obs_shape,
            action_shape,
            horizon,
            device,
            hidden_dim,
            discount,
            init_temperature,
            alpha_lr,
            alpha_beta,
            actor_lr,
            actor_beta,
            actor_log_std_min,
            actor_log_std_max,
            actor_update_freq,
            critic_lr,
            critic_beta,
            critic_tau,
            critic_target_update_freq,
            encoder_type,
            encoder_feature_dim,
            encoder_lr,
            encoder_tau,
            decoder_lr,
            decoder_weight_lambda,
            num_layers,
            num_filters,
            cpc_update_freq,
            log_interval,
            detach_encoder,
            latent_dim,
            data_augs,
            use_metric_loss
        )

        self.train()
        self.critic_target.train()

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':

            batch_obs, batch_action, batch_reward, batch_not_done = replay_buffer.sample_multistep()

            obs = batch_obs[0]
            action = batch_action[0]
            next_obs = batch_obs[1]
            reward = batch_reward[0].unsqueeze(-1)
            not_done = batch_not_done[0].unsqueeze(-1) 

            self.update_critic(obs, action, reward, next_obs, not_done, L, step)

            encoded_batch_obs = []

            for en_iter in range(batch_obs.size(0)):
                encoded_obs = self.critic.encoder(batch_obs[en_iter])
                encoded_batch_obs.append(encoded_obs)
                
            encoded_batch_obs = torch.stack(encoded_batch_obs)   

        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
            self.update_critic(obs, action, reward, next_obs, not_done, L, step)
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )