# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import gpytorch
import torch
import time
from tqdm import tqdm
from tqdm import trange
from torch.autograd import Variable

from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam

from utils.util import save_model

ROOT_MNIST = "./data/"
ROOT_SAVE = "./data/results/"
ROOT_IMAGES = "./data/results/figures/"
ROOT_EXPERIMENTS = "./exp/"

############################################
# TRAIN - SAS METHODS
############################################
def train_sas(model, data_loader, args):
    initial_time = time.time()

    # optimization
    loss_time = []
    loss_curve = []
    with tqdm(range(args.num_epochs)) as pbar:
        for epoch in range(args.num_epochs):
            for batch, data_batch in enumerate(data_loader):
                x_batch = data_batch[0].squeeze().view(-1, model.data_dim)  # N_batch x dim
                x_batch = Variable(x_batch).float()

                if args.amortized == 'yes':
                    loss = model(x_batch)
                elif args.amortized == 'no':
                    ix_batch = data_batch[2]
                    loss = model(x_batch, ix_batch)

                model.optimizer.zero_grad()
                loss.backward()  # Backward pass <- computes gradients
                model.optimizer.step()

                if args.amortized == 'yes':
                    loss_curve.append(model(x_batch).item())
                    loss_time.append(time.time() - initial_time)  # in seconds
                elif args.amortized == 'no':
                    loss_curve.append(model(x_batch, ix_batch).item())
                    loss_time.append(time.time() - initial_time)  # in seconds

            pbar.update()
            pbar.set_description("loss: %.3f" % loss)

    loss_curve = batch * torch.tensor(loss_curve)

    # saving
    if args.save == 'yes':
        save_model(loss_curve, loss_time, model, args, filename=args.filename)

    return loss_curve, loss_time, model


############################################
# TRAIN - BASELINES
############################################
def train_pyro(model,train_loader,args):

    #assert args.dtype == torch.float64, "Training pyro model with dtype 32 - it will crash"
    if args.save == 'yes':
        # pt_name = saving_stuff(None,None,None,args)
        pass

    #### SVI
    optimizer = Adam({"lr": args.lr})
    svi = SVI(model.model, model.guide, optimizer, loss=TraceMeanField_ELBO())

    #### TRAIN
    N = train_loader.dataset.data.shape[0]
    B = train_loader.batch_size
    const = N/B

    losses = list()
    runtimes = list()
    iterator = trange(args.num_epochs)
    start = time.time()
    for i in iterator:
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x,labels,idx in train_loader:

            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x.reshape([args.data_dim,-1]))

        # return epoch loss
        total_epoch_loss_train = const * epoch_loss

        losses.append(total_epoch_loss_train)
        runtimes.append(time.time()-start)

    return




def train_baselines(model, data_loader, args):
    initial_time = time.time()

    # optimization
    loss_time = []
    loss_epoch = []
    loss_curve = []

    if args.baseline in ['svgp', 'mfsvgp']:

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(args.sigma)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}], lr=args.lr)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Training #########
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=args.nof_observations)

        with tqdm(range(args.num_epochs)) as pbar:
            for epoch in range(args.num_epochs):
                for batch, data_batch in enumerate(data_loader):

                    # take care with this when using image data (i.e. MNIST, CIFAR)
                    # cast to float for GPyTorch
                    x_batch = data_batch[0].float() #.squeeze().view(-1, data_loader.dataset.input_dim)  # N_batch x dim
                    y_batch = data_batch[1].float() #.squeeze().view(-1, 1)  # N_batch x 1

                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    loss.sum().backward()
                    optimizer.step()

                    if args.regression == 'yes':
                        loss_epoch.append(loss.sum().item())

                loss_curve.append(sum(loss_epoch) / len(loss_epoch))
                loss_time.append(time.time() - initial_time)  # in seconds
                loss_epoch = []

                pbar.update()
                pbar.set_description("loss: %.3f" % loss.sum())

            loss_curve = torch.tensor(loss_curve)
            loss_time = torch.tensor(loss_time)

        # Saving #########
        if args.save == 'yes':
            save_model(loss_curve, loss_time, model, args, filename=args.filename)

        return loss_curve, loss_time, model, likelihood

        pass
    elif args.baseline == 'meanfield_svgp':
        pass
    else:
        pass