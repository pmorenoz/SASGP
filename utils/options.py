
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import numpy as np
import random
import torch
import gpytorch

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from models.gplvm import SASGP, BayesianSASGP
from models.gplvm import NonAmortizedSASGP, NonAmortizedBayesianSASGP


from utils.trainers import train_sas, train_baselines
# from utils.evaluation import evidence_sas, evidence_baseline
from utils.plotting import plot_sas, plot_baseline, plot_prediction
from utils.util import create_directory, getDate, getDateTime, getTime
from utils import data_loaders

import glob
import pickle
import datetime

ROOT_SAVE = "./data/results/"
ROOT_IMAGES = "./data/results/figures/"
ROOT_EXPERIMENTS = "./exp/"

def select_model(args, data_dimension=784):

    model = None
    if args.model == 'sas':
        # Kernel + Likelihood Setup --
        kernel_ls = torch.tensor(args.kernel_ls, requires_grad=True)
        kernel_a = torch.tensor(args.kernel_a, requires_grad=True)
        sigma = torch.tensor(args.sigma, requires_grad=True)

        kernel = RBF(length_scale=kernel_ls, variance=kernel_a, jitter=args.jitter, input_dim=args.latent_dim, ARD=args.ard, fit_hyp=True)
        likelihood = Gaussian(sigma=sigma, fit_noise=True)

        if args.amortized == 'yes':
            if args.inference == 'gplvm':
                model = SASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                                      latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations)
            elif args.inference == 'bayesian':
                model = BayesianSASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                                      latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations)
            else:
                raise NotImplementedError

        elif args.amortized == 'no':
            if args.inference == 'gplvm':
                model = NonAmortizedSASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                              latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations, dataset=args.dataset)
            elif args.inference == 'bayesian':
                model = NonAmortizedBayesianSASGP(kernel, likelihood, learning_rate=args.lr, active_set=args.num_active,
                                      latent_dim=args.latent_dim, data_dim=data_dimension, data_size=args.nof_observations)
            else:
                raise NotImplementedError

    elif args.model == 'pyro':
        pass

    else:
        raise NotImplementedError

    return model

def train(model, data_loader, args):

    # Dictionary with metrics as method output.
    metrics_dict = {}

    if args.model == 'sas':
        loss, time, model = train_sas(model, data_loader, args)

        metrics_dict['loss'] = loss
        metrics_dict['time'] = time

        # Plotting of the training curves.
        if args.plot == 'yes':
            plot_sas(loss, time, model, data_loader, args)

        if 'toy' in args.dataset: # generated data where we have ground truth
            print('\n---------------------------------------------------------')
            print('[true] Likelihood sigma =', args.true_sigma)
            print('[true] Length-scale =', args.true_ls)
            print('[true] Amplitude =', args.true_a)

        print('---------------------------------------------------------')
        print('Likelihood sigma =', torch.exp(model.likelihood.log_sigma).item())
        print('Length-scales =', [torch.exp(ls).item() for ls in model.kernel.log_length_scale])
        print('Amplitude =', torch.exp(model.kernel.log_variance).item())
        print('---------------------------------------------------------')

        return model, metrics_dict

    elif args.model == 'baseline':
        loss, time, model, lik = train_baselines(model, data_loader, args)

        metrics_dict['loss'] = loss
        metrics_dict['time'] = time

        # Plotting of the training curves.
        if args.plot == 'yes':
            plot_baseline(loss, time, model, lik, data_loader, args)
        
        if 'toy' in args.dataset: # generated data where we have ground truth
            print('\n---------------------------------------------------------')
            print('[true] Likelihood sigma =', args.true_sigma)
            print('[true] Length-scale =', args.true_ls)
            print('[true] Amplitude =', args.true_a)

        print('---------------------------------------------------------')
        print('Likelihood sigma =', lik.noise.detach().item())
        print('Length-scale =', [ls.item() for ls in model.covar_module.base_kernel.lengthscale.detach().flatten()])
        print('Amplitude =', model.covar_module.outputscale.detach().item())
        print('---------------------------------------------------------')

        return [model, lik], metrics_dict
    else:
        raise NotImplementedError


def save(model, args, metrics):

    print(metrics)

    if args.save:
        # Complete dictionary with model parameters + arg parser.
        metrics['args'] = args

        save_dir = ROOT_EXPERIMENTS + str(args.experiment)
        create_directory(save_dir)

        if args.filename == 'none':
            file_dir = save_dir + '/' + getDateTime() + '_' + str(args.seed) + '.pt'
        else:
            file_dir = save_dir + '/' + args.filename + '.pt'
        with open(file_dir, 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return file_dir

    else:
        print('Experiment was not saved --')
