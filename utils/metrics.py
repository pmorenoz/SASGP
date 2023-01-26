# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Pablo Moreno-Muñoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import torch
import math
import numpy as np

##################################################
#   RMSE -- Root Mean Square Error
##################################################
def rmse(y_star, m_star):
    #y_fold = y_star.view(y_star.shape[0], -1)
    rmse = torch.sqrt(torch.mean((y_star - m_star)**2))
    return rmse

##################################################
#   MAE -- Mean Absolute Error
##################################################
def mae(y_star, m_star):
    #y_fold = y_star.view(y_star.shape[0], -1)
    mae = torch.mean(torch.abs(y_star - m_star))
    return mae

##################################################
#   NLPD -- Negative Log-Predictive Density
##################################################
def nlpd(y_star, m_star, v_star):
    #y_fold = y_star.view(y_star.shape[0], -1)
    try:
        nlpd = torch.abs(0.5*math.log(2*math.pi) + 0.5*torch.mean(torch.log(v_star) + (y_star - m_star)**2/v_star))
    except:
        nlpd = np.nan

    return nlpd

##################################################
#   Error Metrics for GPs
#   1. RMSE
#   2. MAE
#   3. NLPD
##################################################
def error_metrics(y_star, m_star, v_star):
    rmse = rmse(y_star, m_star)
    mae = mae(y_star, m_star)
    nlpd = nlpd(y_star, m_star, v_star)
    return rmse, mae, nlpd
