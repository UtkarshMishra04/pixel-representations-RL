import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F

import gym

import os
from collections import deque
import random
import math
import time

from gym import spaces

def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        if isinstance(output_mod, str):
            if output_mod == 'tanh':
                output_mod = torch.nn.Tanh()
            else:
                assert False
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def conv_mlp_encoder(input_shape, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Conv2d(input_shape[0], 32, 3, stride=1)]
    else:
        mods = [nn.Conv2d(input_shape[0], 32, 3, stride=1), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(inplace=True)]
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def conv_mlp_decoder(output_shape, feature_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        assert False
    else:
        pads = [0, 1, 0]
        mods = [nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=1), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            output_padding = pads[i]
            mods += [nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=output_padding), nn.ReLU(inplace=True)]
        mods.append(nn.ConvTranspose2d(32, output_shape[0], 3, stride=2, output_padding=1))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

import numpy as np

# https://pswww.slac.stanford.edu/svn-readonly/psdmrepo/RunSummary/trunk/src/welford.py
class Welford(object):
    """Knuth implementation of Welford algorithm.
    """

    def __init__(self, x=None):
        self._K = np.float64(0.)
        self.n = np.float64(0.)
        self._Ex = np.float64(0.)
        self._Ex2 = np.float64(0.)
        self.shape = None
        self._min = None
        self._max = None
        self._init = False
        self.__call__(x)

    def add_data(self, x):
        """Add data.
        """
        if x is None:
            return

        x = np.array(x)
        self.n += 1.
        if not self._init:
            self._init = True
            self._K = x
            self._min = x
            self._max = x
            self.shape = x.shape
        else:
            self._min = np.minimum(self._min, x)
            self._max = np.maximum(self._max, x)

        self._Ex += (x - self._K) / self.n
        self._Ex2 += (x - self._K) * (x - self._Ex)
        self._K = self._Ex

    def __call__(self, x):
        self.add_data(x)

    def max(self):
        """Max value for each element in array.
        """
        return self._max

    def min(self):
        """Min value for each element in array.
        """
        return self._min

    def mean(self, axis=None):
        """Compute the mean of accumulated data.

           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float64(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val

    def sum(self, axis=None):
        """Compute the sum of accumulated data.
        """
        return self.mean(axis=axis)*self.n

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return  np.zeros(self.shape)

        val = np.array((self._Ex2 - (self._Ex*self._Ex)/np.float64(self.n)) / np.float64(self.n-1.))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))

