# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Pablo Moreno-Munoz 
# Dept. of Signal Processing and Com. -- (pmoreno@tsc.uc3m.es, pabmo@dtu.dk)
# Universidad Carlos III de Madrid


import torch
import numpy as np
from utils.math import squared_distance

class Kernel(torch.nn.Module):
    """
    -- * Kernel

        Description:        Base class for kernels
        ----------
        Parameters
        ----------
        - input_dim:        int / dimensionality of X
    """
    def __init__(self, input_dim=None):
        super(Kernel, self).__init__()

        # Input dimension -- x
        if input_dim is None:
            input_dim = 1
        else:
            input_dim = int(input_dim)

        self.input_dim = input_dim

