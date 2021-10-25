# Finite Trajectory Similarity Code
"""Transition dynamics for the agent in original MDP."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import multistep_utils as utils

import numpy as np

from torch.nn import BatchNorm1d

import time

class MultiStepDynamicsModel(nn.Module):
    def __init__(self,
                 obs_dim, 
                 action_dim, 
                 xu_enc_hidden_dim, 
                 x_dec_hidden_dim,  
                 rec_latent_dim, 
                 rec_num_layers,
                 clip_grad_norm=0.2,
                 xu_enc_hidden_depth=2,
                 x_dec_hidden_depth=2,
                 rec_type='LSTM'
                 ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_grad_norm = clip_grad_norm

        # Manually freeze the goal locations
        self.freeze_dims = None

        self.rec_type = rec_type
        self.rec_num_layers = rec_num_layers
        self.rec_latent_dim = rec_latent_dim

        self.xu_enc = utils.mlp(
            obs_dim+action_dim, xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth)
        self.x_dec = utils.mlp(
            rec_latent_dim, x_dec_hidden_dim, obs_dim, x_dec_hidden_depth)

        self.apply(utils.weight_init) # Don't apply this to the recurrent unit.

        mods = [self.xu_enc, self.x_dec]

        if rec_num_layers > 0:
            if rec_type == 'LSTM':
                self.rec = nn.LSTM(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            elif rec_type == 'GRU':
                self.rec = nn.GRU(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            else:
                assert False
            mods.append(self.rec)

        params = utils.get_params(mods)

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.rec.flatten_parameters()

    def init_hidden_state(self, init_x):
        assert init_x.dim() == 2
        n_batch = init_x.size(0)

        if self.rec_type == 'LSTM':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == 'GRU':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
        else:
            assert False

        return h

    def unroll(self, x, us, detach_xt=False):
        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)
        assert us.size(1) == n_batch

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(x)

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]

            if detach_xt:
                xt = xt.detach()

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if t==0:
                xu_emb_1 = xu_emb.squeeze(0)
            if self.rec_num_layers > 0:
                xtp1_emb, h = self.rec(xu_emb, h)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))
            pred_xs.append(xtp1)
            xt = xtp1

        pred_xs = torch.stack(pred_xs)

        h_states, _ = h

        return pred_xs, h_states[-1]

    def forward(self, x, us):
        return self.unroll(x, us)

class MultiStepPixelDynamicsModel(nn.Module):
    def __init__(self,
                 obs_shape, 
                 action_shape,
                 obs_hidden_dim, 
                 xu_enc_hidden_dim, 
                 x_dec_hidden_dim,  
                 rec_latent_dim, 
                 rec_num_layers,
                 clip_grad_norm=0.2,
                 xu_enc_hidden_depth=2,
                 x_dec_hidden_depth=2,
                 rec_type='LSTM'
                 ):
        super().__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.obs_hidden_dim = obs_hidden_dim
        self.clip_grad_norm = clip_grad_norm

        # Manually freeze the goal locations
        self.freeze_dims = None

        self.rec_type = rec_type
        self.rec_num_layers = rec_num_layers
        self.rec_latent_dim = rec_latent_dim

        self.obs_enc = utils.conv_mlp_encoder(
            obs_shape, obs_hidden_dim, 2)
        self.obs_enc_fc = nn.Linear(32 * 80 * 80, obs_hidden_dim)
        self.obs_enc_ln = nn.LayerNorm(obs_hidden_dim)
        self.xu_enc = utils.mlp(
            obs_hidden_dim+action_shape[0], xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth)
        self.x_dec = utils.mlp(
            rec_latent_dim, x_dec_hidden_dim, obs_hidden_dim, x_dec_hidden_depth)
        self.obs_dec_fc = nn.Linear(obs_hidden_dim, 32 * 9 * 9)
        self.obs_dec = utils.conv_mlp_decoder(
            obs_shape, obs_hidden_dim, 2)

        self.latent_fc = utils.mlp(
            rec_latent_dim+action_shape[0], xu_enc_hidden_dim, rec_latent_dim, xu_enc_hidden_depth)

        self.apply(utils.weight_init) # Don't apply this to the recurrent unit.

        mods = [self.obs_enc, self.obs_enc_fc, self.obs_enc_ln, self.xu_enc, self.x_dec, self.obs_dec_fc, self.obs_dec, self.latent_fc]

        if rec_num_layers > 0:
            if rec_type == 'LSTM':
                self.rec = nn.LSTM(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            elif rec_type == 'GRU':
                self.rec = nn.GRU(
                    rec_latent_dim, rec_latent_dim, num_layers=rec_num_layers)
            else:
                assert False
            mods.append(self.rec)

        params = utils.get_params(mods)

    def __getstate__(self):
        d = self.__dict__
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.rec.flatten_parameters()

    def init_hidden_state(self, init_x):
        n_batch = init_x.size(0)

        if self.rec_type == 'LSTM':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
            c = torch.zeros_like(h)
            h = (h, c)
        elif self.rec_type == 'GRU':
            h = torch.zeros(
                self.rec_num_layers, n_batch, self.rec_latent_dim, device=init_x.device)
        else:
            assert False

        return h

    def unroll(self, x, us, detach_xt=False):
        n_batch = x.size(0)

        if self.rec_num_layers > 0:
            h = self.init_hidden_state(x)

        pred_xs = []
        xt = x
        for t in range(us.size(0)):
            ut = us[t]
        
            if detach_xt:
                xt = xt.detach()

            xt = self.obs_enc(xt / 255)
            xt = xt.view(xt.size(0), -1)
            xt = self.obs_enc_fc(xt)
            xt = torch.relu(xt)
            xt = self.obs_enc_ln(xt)

            xut = torch.cat((xt, ut), dim=1)
            xu_emb = self.xu_enc(xut).unsqueeze(0)
            if t==0:
                xu_emb_1 = xu_emb.squeeze(0)
            if self.rec_num_layers > 0:
                h_tm1 = h
                xtp1_emb, h = self.rec(xu_emb, h_tm1)
            else:
                xtp1_emb = xu_emb
            xtp1 = xt + self.x_dec(xtp1_emb.squeeze(0))

            xtp1_dec = self.obs_dec_fc(xtp1)

            xtp1_dec = self.obs_dec(xtp1_dec.view(-1, 32, 9, 9))

            pred_xs.append(xtp1_dec)
            xt = xtp1_dec

        pred_xs = torch.stack(pred_xs)

        h_tm1_states, _ = h_tm1
        h_states, _ = h

        hpred_t = self.latent_fc(torch.cat((h_tm1_states[-2],ut),dim=1))

        return pred_xs, h_states[-1], hpred_t

    def forward(self, x, us):
        return self.unroll(x, us)