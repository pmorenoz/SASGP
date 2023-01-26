# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Pablo Moreno-Munoz 
# Dept. of Signal Processing and Com. -- (pmoreno@tsc.uc3m.es, pabmo@dtu.dk)
# Universidad Carlos III de Madrid

import torch
import numpy as np
from utils.math import squared_distance
from kernels.kernel import Kernel

class Stationary(Kernel):
    """
    -- * Stationary

        Description:        Class for Stationary Kernel
        ----------
        Parameters
        ----------
        - length_scale:     float / lengthscale hyperparameter
        - variance:         float / variance hyperparameter
        - input_dim:        int / dimensionality of X
        - ARD:              bool / automatic relevant determination? a pair of hyperparameters per dim of X
        - fit_hyp:          bool / trainable hyperparams?
    """

    def __init__(self, variance=None, length_scale=None, input_dim=None, ARD=False, fit_hyp=True):
        super().__init__(input_dim)

        if input_dim is None:
            self.input_dim = 1
        else:
            self.input_dim = input_dim

        self.ARD = ARD  # Automatic relevance determination
        # Length-scale/smoothness of the kernel -- l
        if self.ARD:
            if length_scale is None:
                log_ls = torch.log(torch.tensor(0.1)) * torch.ones(self.input_dim)
            else:
                log_ls = torch.log(length_scale) * torch.ones(self.input_dim)
        else:
            if length_scale is None:
                log_ls = torch.log(torch.tensor(0.1)) * torch.ones(1)
            else:
                log_ls = torch.log(length_scale) * torch.ones(1)


        # Variance/amplitude of the kernel - /sigma
        if variance is None:
            log_variance = torch.log(torch.tensor(2.0))
        else:
            log_variance = torch.log(variance)

        if fit_hyp:
            self.log_length_scale = torch.nn.Parameter(log_ls, requires_grad=True)
            self.log_variance = torch.nn.Parameter(log_variance*torch.ones(1), requires_grad=True)
            self.register_parameter('length_scale', self.log_length_scale)
            self.register_parameter('variance', self.log_variance)
        else:
            self.log_length_scale = torch.nn.Parameter(log_ls, requires_grad=fit_hyp)
            self.log_variance = torch.nn.Parameter(log_variance*torch.ones(1), requires_grad=fit_hyp)
            self.register_parameter('length_scale', self.log_length_scale)
            self.register_parameter('variance', self.log_variance)

    def squared_dist(self, X, X2):
        """
        Returns the SCALED squared distance between X and X2.
        """
        length_scale = torch.exp(self.log_length_scale).abs().clamp(min=0.01, max=10.0)  # minimum enforced

        if not self.ARD:
            if X2 is None:
                dist = squared_distance(X/length_scale)
            else:
                dist = squared_distance(X/length_scale, X2/length_scale)
        else:
            if X2 is None:
                dist = squared_distance(X / length_scale)
            else:
                dist = squared_distance(X / length_scale, X2 / length_scale)

        return dist

    def Kdiag(self, X):
        variance = torch.abs(torch.exp(self.variance))
        return variance.expand(X.size(0))
