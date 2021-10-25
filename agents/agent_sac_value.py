import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
import data_augs as rad
from agents.auxiliary_funcs import BaseSacAgent
from multistep_dynamics import MultiStepDynamicsModel

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
        print('################ Starting Case 5: Baseline + Value Aware ###############')
        print('###################### Value: Yes; Transition: Yes #####################')
        print('################### Value decoder through Transition ###################')
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

        if self.encoder_type == 'pixel':

            self.transition_model = MultiStepDynamicsModel(
                                                    obs_dim=encoder_feature_dim, 
                                                    action_dim=action_shape[0],
                                                    xu_enc_hidden_dim=128, 
                                                    x_dec_hidden_dim=128,  
                                                    rec_latent_dim=128, 
                                                    rec_num_layers=1,
                                                ).to(device)

            self.reward_decoder = nn.Sequential(
                nn.Linear(encoder_feature_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.transition_model.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    

    def compute_value_targets(self, rewards, value_terminal_state):

        value_targets = []
        for batch_iter in range(rewards.size(1)):
            value = 0
            for hstep in range(rewards.size(0)):
                value += self.discount**hstep * rewards[hstep, batch_iter]
            value += value_terminal_state[batch_iter][0]
            value_targets.append([value])

        return torch.Tensor(value_targets)

    def update_transition_model(self, obs, action, targets, rewards, target_value, next_obs):
        
        transition_next_states, _ = self.transition_model.forward(obs, action.unsqueeze(0))
        transition_next_states = transition_next_states.squeeze()

        _, policy_action, log_pi, _ = self.actor(next_obs)
        Q1, Q2 = self.critic(transition_next_states, policy_action.detach(), noencode=True)
        pred_value = torch.min(Q1,Q2) - self.alpha.detach() * log_pi

        loss = 0
        reward_loss = nn.MSELoss()(pred_value, target_value)

        return loss, reward_loss

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':

            batch_obs, batch_action, batch_reward, batch_not_done = replay_buffer.sample_multistep()

            obs = batch_obs[0]
            action = batch_action[0]
            next_obs = batch_obs[1]
            reward = batch_reward[0].unsqueeze(-1)
            not_done = batch_not_done[0].unsqueeze(-1) 

            self.update_critic(obs, action, reward, next_obs, not_done, L, step)

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

            with torch.no_grad():
                _, policy_action, log_pi, _ = self.actor(next_obs)
                Q1, Q2 = self.critic(next_obs, policy_action)
                target_value = torch.min(Q1, Q2) - self.alpha.detach() * log_pi

            encoded_batch_obs = []

            for en_iter in range(batch_obs.size(0)):
                encoded_obs = self.critic.encoder(batch_obs[en_iter])
                encoded_batch_obs.append(encoded_obs)
                
            encoded_batch_obs = torch.stack(encoded_batch_obs)
            target_rewards = batch_reward[:-1]

            input_encoded_obs = encoded_batch_obs[0]
            target_encoded_obs = encoded_batch_obs[1].detach()

            transition_loss, reward_loss = self.update_transition_model(input_encoded_obs, action, target_encoded_obs, target_rewards, target_value.detach(), next_obs)
                
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        if step % self.log_interval == 0:
            L.log('train_critic/reward_loss', reward_loss, step)
            L.log('train_critic/transition_loss', transition_loss, step)

        # Optimize the critic, encoder and transition
        total_loss = transition_loss + reward_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()