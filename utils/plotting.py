

# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)


from models.gplvm import SASGP, BayesianSASGP
from utils.data_loaders import all_data

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd import Variable
import numpy as np

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tikzplotlib import save as tikz_save

font = {'family' : 'serif',
        'size'   : 22}

plt.rc('text', usetex=True)
plt.rc('font', **font)

ROOT_MNIST = "./data/"
ROOT_SAVE = "./data/results/"
ROOT_IMAGES = "./data/results/figures/"
ROOT_EXPERIMENTS = "./exp/"

# COOLORS.CO palettes
color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']
color_palette_6 = ['#21295c','#1b3b6f','#065a82','#1c7293','#9eb3c2']
color_palette_7 = ['#f7b267','#f79d65','#f4845f','#f27059','#f25c54']
color_palette_10 = ['#001219','#005F73','#0A9396','#94D2BD','#E9D8A6','#EE9B00','#CA6702','#BB3E03','#AE2012','#9B2226']

palette_red = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
palette_blue = ["#012a4a","#013a63","#01497c","#014f86","#2a6f97","#2c7da0","#468faf","#61a5c2","#89c2d9","#a9d6e5"]
palette_green = ['#99e2b4','#88d4ab','#78c6a3','#67b99a','#56ab91','#469d89','#358f80','#248277','#14746f','#036666']
palette_pink = ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf","#6d23b6","#6411ad","#571089","#47126b"]
palette_super_red = ["#641220","#6e1423","#85182a","#a11d33","#a71e34","#b21e35","#bd1f36","#c71f37","#da1e37","#e01e37"]

palettes = [palette_red, palette_pink, palette_blue, palette_green]

palette_9_colors = ['#54478C', '#2C699A', '#048BA8', '#0DB39E', '#16DB93', '#83E377', '#B9E769', '#EFEA5A', '#F1C453', '#F29E4C']
palette_20_colors = ["#001219","#005f73","#0a9396","#94d2bd","#e9d8a6","#ee9b00","#ca6702","#bb3e03","#ae2012","#9b2226","#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]

palette = palette_20_colors

font = {'family' : 'serif',
        'size'   : 35}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def get_experiment_name(args):
    model = args.model
    if args.model == 'baseline':
        model += f'_{args.baseline}'
    name = f'{args.dataset}_model={model}_a={args.num_active}_bs={args.batch_size}_seed={args.seed}_kls={args.kernel_ls}_kvar={args.kernel_a}_sigma={args.sigma}' 
    if args.experiment != '':
        name += f'_{args.experiment}'
    return name


def plot_prediction(f_star, v_star, x_star, y_star, model, data_loader, args):

    #####################################################
    # PLOTTING CONFIDENCE INTERVALS FOR TEST DATA
    #####################################################
    fig, ax = plt.subplots(figsize=(12, 8))
    x_star, order_ix = torch.sort(x_star, dim=0)
    f_star = f_star[order_ix.squeeze()].squeeze()
    v_star = v_star[order_ix.squeeze()].squeeze()
    y_star = y_star[order_ix.squeeze()].squeeze()

    # Plotting Training Data
    x, y = None, None
    all_data_loader = all_data(data_loader)
    for _, data in enumerate(all_data_loader):
        x = data[0].squeeze().view(-1, data_loader.dataset.input_dim)  # N x dim
        x = Variable(x).float()

        y = data[1].squeeze().view(-1, 1)  # N x 1
        y = Variable(y).float()

    plt.plot(x, y, color=palette[3], ls='', marker='x', lw=2.0, alpha=0.8)

    # Plotting Test Data
    plt.plot(x_star, y_star, color=palette[2], ls='', marker='o', ms=3.0, lw=2.0, alpha=0.8)

    upper_f_star = f_star + 2*torch.sqrt(v_star)
    lower_f_star = f_star - 2*torch.sqrt(v_star)

    plt.plot(x_star.detach().numpy(), f_star.detach().numpy(), color=palette[0], lw=2.0, alpha=0.8)
    plt.plot(x_star.detach().numpy(), upper_f_star.detach().numpy(), color=palette[0], lw=2.0, alpha=0.8)
    plt.plot(x_star.detach().numpy(), lower_f_star.detach().numpy(), color=palette[0], lw=2.0, alpha=0.8)

    ax.set_xlim(left=torch.min(x_star), right=torch.max(x_star))
    plt.title(r'SAS - $\mathcal{GP}$  - \textsc{' + 'regression' + '}')
    plt.ylabel(r'Output $\bm{y}$')
    plt.xlabel(r'Input $\bm{x}$')
    if args.model == 'sas':
        plt.savefig(fname=ROOT_IMAGES + args.experiment + '_A' + str(args.num_active) + '_prediction.pdf', format='pdf',
                    bbox_inches='tight')
    else:
        plt.savefig(fname=ROOT_IMAGES + args.experiment + '_A' + str(args.num_active) + '_prediction' + args.baseline +'.pdf', format='pdf',
                    bbox_inches='tight')


def plot_baseline(loss, time, model, lik, data_loader, args, time_axis=True):
    ############################################
    # PLOTTING LOSS CURVE
    ############################################

    fig, ax = plt.subplots(figsize=(12, 8))
    time = time / 60.0  # minutes

    if time_axis:
        plt.plot(time, -loss, color=palette[0], lw=2.0, alpha=0.8)
    else:
        plt.plot(-loss, color=palette[0], lw=2.0, alpha=0.8)

    if args.marginal_lik == 'yes':
        x, y = None, None
        all_data_loader = all_data(data_loader)
        for _, data in enumerate(all_data_loader):
            x = data[0].squeeze().view(-1, data_loader.dataset.input_dim)  # N x dim
            x = Variable(x).float()

            y = data[1].squeeze().view(-1, 1)  # N x 1
            y = Variable(y).float()

            # Model's Negative log-marginal likelihood (NLML)
            lik_sigma = lik.noise.detach()
            kernel_ls = model.covar_module.base_kernel.lengthscale.detach()
            kernel_a =  model.covar_module.outputscale.detach()
            trained_kernel = RBF(length_scale=kernel_ls, variance=kernel_a, jitter=args.jitter, input_dim=args.data_dimension,
                              ARD=True, fit_hyp=True)
            trained_likelihood = Gaussian(sigma=lik_sigma, fit_noise=True)
            trained_model = SASGPR(trained_kernel, trained_likelihood, learning_rate=args.lr, active_set=args.num_active,
                                data_size=args.nof_observations)
            nlml = trained_model.exact_marginal_likelihood(x, y).detach().numpy()
            plt.plot([0,len(loss)],[-nlml,-nlml], color='r', lw=2.0, alpha=0.8)

            experiment_name = get_experiment_name(args)
            print('Exact LML is:', -nlml)
            np.savetxt("data/exact_lml/" + experiment_name + '.txt', np.array([-nlml]))

            if args.dataset == 'toy_regression':
                # Ground Truth Negative log-marginal likelihood (NLML)
                true_kernel_ls = torch.tensor(args.true_ls, requires_grad=False)
                true_kernel_a = torch.tensor(args.true_a, requires_grad=False)
                true_sigma = torch.tensor(args.true_sigma, requires_grad=False)
                true_kernel = RBF(length_scale=true_kernel_ls, variance=true_kernel_a, jitter=args.jitter, input_dim=1,
                                ARD=True, fit_hyp=True)
                true_likelihood = Gaussian(sigma=true_sigma, fit_noise=True)
                true_model = SASGPR(true_kernel, true_likelihood, learning_rate=args.lr, active_set=args.num_active,
                                    data_size=args.nof_observations)

                nlml_gt = true_model.exact_marginal_likelihood(x, y).detach().numpy()
                plt.plot([0, len(loss)], [-nlml_gt, -nlml_gt], color='r', ls='--', lw=2.0, alpha=0.8)

    plt.title(r'SAS - $\mathcal{GP}$  - \textsc{' + 'regression' + '}')
    plt.ylabel(r'$\mathcal{L}$')

    if time_axis:
        plt.xlabel('Run Time [minutes]')
    else:
        plt.xlabel('Run Time [epochs]')

    ax.set_xlim(left=time[0], right=time[-1])
    experiment_name = get_experiment_name(args)
    plt.savefig(fname=ROOT_IMAGES + experiment_name + '_A' + str(args.num_active) + '_loss_' + args.baseline +'.pdf', format='pdf', bbox_inches='tight')


def plot_sas(loss, time, model, data_loader, args, time_axis=True):
    ############################################
    # PLOTTING LOSS CURVE
    ############################################

    fig, ax = plt.subplots(figsize=(12, 8))
    time = time/60.0 #minutes

    if time_axis:
        plt.plot(time, -loss, color=palette[0], lw=2.0, alpha=0.8)
    else:
        plt.plot(-loss, color=palette[0], lw=2.0, alpha=0.8, label='train_lml')

    if args.marginal_lik == 'yes':
        x, y = None, None
        all_data_loader = all_data(data_loader)
        for _, data in enumerate(all_data_loader):
            x = data[0].squeeze().view(-1, data_loader.dataset.input_dim)  # N x dim
            x = Variable(x).float()

            y = data[1].squeeze().view(-1, 1)  # N x 1
            y = Variable(y).float()

        # Model's Negative log-marginal likelihood (NLML)
        nlml = model.exact_marginal_likelihood(x, y).detach().numpy()
        plt.plot([0,len(loss)],[-nlml,-nlml], color='r', lw=2.0, alpha=0.8, label ='exact LML')

        experiment_name = get_experiment_name(args)
        print('Exact LML is:', -nlml)
        np.savetxt("data/exact_lml/" + experiment_name + '.txt', np.array([-nlml]))

        if args.dataset == 'toy_regression':
            # Ground Truth Negative log-marginal likelihood (NLML)
            true_kernel_ls = torch.tensor(args.true_ls, requires_grad=False)
            true_kernel_a = torch.tensor(args.true_a, requires_grad=False)
            true_sigma = torch.tensor(args.true_sigma, requires_grad=False)
            true_kernel = RBF(length_scale=true_kernel_ls, variance=true_kernel_a, jitter=args.jitter, input_dim=1, ARD=True, fit_hyp=True)
            true_likelihood = Gaussian(sigma=true_sigma, fit_noise=True)
            true_model = SASGPR(true_kernel, true_likelihood, learning_rate=args.lr, active_set=args.num_active, data_size=args.nof_observations)

            nlml_gt = true_model.exact_marginal_likelihood(x, y).detach().numpy()
            plt.plot([0,len(loss)],[-nlml_gt,-nlml_gt], color='r', ls='--', lw=2.0, alpha=0.8, label='ground truth LML')

    plt.legend()
    plt.title(r'SAS - $\mathcal{GP}$  - \textsc{' + 'regression' + '}')
    plt.ylabel(r'$\mathcal{L}$')
    if time_axis:
        plt.xlabel('Run Time [minutes]')
    else:
        plt.xlabel('Run Time [epochs]')
    ax.set_xlim(left=time[0], right=time[-1])
    experiment_name = get_experiment_name(args)
    plt.savefig(fname=ROOT_IMAGES + experiment_name + '_loss.pdf', format='pdf', bbox_inches='tight')

def plot_curves(loss_curve, time_curve, path, filename='loss_curve'):
    subsampling_rate = 10 # one of every 100 data points
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.asarray(time_curve)[::subsampling_rate] / 3600, -np.asarray(loss_curve)[::subsampling_rate], 'b', alpha=0.5, lw=2.0)
    plt.xlabel('Loss') # would be log-marginal likelihood for GPLVM models and ELBO for Bayesian ones
    plt.xlabel('Run Time [hours]')
    plt.savefig(fname=path + filename + '_loss_curve.pdf', format='pdf')


def plot_mnist_colors(data_loader, model, path, filename='latent_x_color'):
    fig, ax = plt.subplots(figsize=(16, 16))
    max_test_data = 1000

    z_0_lim_min, z_0_lim_max = -1.0, 1.0
    z_1_lim_min, z_1_lim_max = -1.0, 1.0

    # Plot point-estimates x or predictive distribution according to type of model
    if isinstance(model, SASGP):

        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                labels = x_test[1]

                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.amortization_net(x_test)

                for n, x_n in enumerate(x_test):
                    ax.plot(z_test[n, 0].detach().numpy().flatten(),
                            z_test[n, 1].detach().numpy().flatten(),
                            'o', ms=15, alpha=0.75,
                            color=palette_9_colors[labels[n]])

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    elif isinstance(model, BayesianSASGP):
        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                labels = x_test[1]

                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.mu_z(x_test)

                for n, x_n in enumerate(x_test):
                    ax.plot(z_test[n, 0].detach().numpy().flatten(),
                            z_test[n, 1].detach().numpy().flatten(),
                            'o', ms=15, alpha=0.75,
                            color=palettes[labels[n]][0])

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    plt.savefig(fname=path + filename + '_color.pdf', format='pdf')



def plot_mnist_images(data_loader, model, path, filename='latent_x', max_test_data=1000):
    img_w, img_h = 28, 28
    zoom = 1.2
    fig, ax = plt.subplots(figsize=(16, 16))

    z_0_lim_min, z_0_lim_max = -1.0, 1.0
    z_1_lim_min, z_1_lim_max = -1.0, 1.0

    # Plot point-estimates x or predictive distribution according to type of model
    if isinstance(model, SASGP):

        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.amortization_net(x_test)

                for n, x_n in enumerate(x_test):
                    image = x_n.detach().numpy().reshape((img_w, img_h))
                    im = OffsetImage(image, zoom=zoom, cmap=plt.cm.gray)
                    ab = AnnotationBbox(im, (z_test[n, 0].detach().numpy().flatten(),
                                             z_test[n, 1].flatten().detach().numpy().flatten()),
                                        xycoords='data', frameon=False)
                    ax.add_artist(ab)

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    elif isinstance(model, BayesianSASGP):
        print('here')
        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.mu_z(x_test)

                for n, x_n in enumerate(x_test):
                    image = x_n.detach().numpy().reshape((img_w, img_h))
                    im = OffsetImage(image, zoom=zoom, cmap=plt.cm.gray)
                    ab = AnnotationBbox(im, (z_test[n, 0].detach().numpy().flatten(),
                                             z_test[n, 1].flatten().detach().numpy().flatten()),
                                        xycoords='data', frameon=False)
                    ax.add_artist(ab)

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    plt.savefig(fname=path + filename + '_images.pdf', format='pdf')


def plot_mnist_images(data_loader, model, path, filename='latent_x', max_test_data=1000):
    img_w, img_h = 28, 28
    zoom = 1.2
    fig, ax = plt.subplots(figsize=(16, 16))

    z_0_lim_min, z_0_lim_max = -1.0, 1.0
    z_1_lim_min, z_1_lim_max = -1.0, 1.0

    # Plot point-estimates x or predictive distribution according to type of model
    if isinstance(model, SASGP):

        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.amortization_net(x_test)

                for n, x_n in enumerate(x_test):
                    image = x_n.detach().numpy().reshape((img_w, img_h))
                    im = OffsetImage(image, zoom=zoom, cmap=plt.cm.gray)
                    ab = AnnotationBbox(im, (z_test[n, 0].detach().numpy().flatten(),
                                             z_test[n, 1].flatten().detach().numpy().flatten()),
                                        xycoords='data', frameon=False)
                    ax.add_artist(ab)

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    elif isinstance(model, BayesianSASGP):
        print('here')
        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.mu_z(x_test)

                for n, x_n in enumerate(x_test):
                    image = x_n.detach().numpy().reshape((img_w, img_h))
                    im = OffsetImage(image, zoom=zoom, cmap=plt.cm.gray)
                    ab = AnnotationBbox(im, (z_test[n, 0].detach().numpy().flatten(),
                                             z_test[n, 1].flatten().detach().numpy().flatten()),
                                        xycoords='data', frameon=False)
                    ax.add_artist(ab)

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    plt.savefig(fname=path + filename + '_images.pdf', format='pdf')

def plot_reuters_colors(data_loader, labels, model, path, filename='latent_x_color'):
    fig, ax = plt.subplots(figsize=(16, 16))

    data_dimension = 2000
    z_0_lim_min, z_0_lim_max = -1.0, 1.0
    z_1_lim_min, z_1_lim_max = -1.0, 1.0

    # Plot point-estimates x or predictive distribution according to type of model
    if isinstance(model, SASGP):
        for i, x_test in enumerate(data_loader):

            x_test = x_test[0].squeeze().view(-1, data_dimension)  # 1 x dim
            x_test = Variable(x_test).float()
            z_test = model.amortization_net(x_test)

            for n, x_n in enumerate(x_test):
                ax.plot(z_test[n, 0].detach().numpy().flatten(),
                        z_test[n, 1].detach().numpy().flatten(),
                        'o', ms=8, alpha=0.5,
                        color=palette_20_colors[int(labels[n])])

                if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                    z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                    z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                    z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                    z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min-0.5, z_0_lim_max +0.5])
        plt.ylim([z_1_lim_min-0.5, z_1_lim_max +0.5])

    elif isinstance(model, BayesianSASGP):
        print('here_color')
        for i, x_test in enumerate(data_loader):

            x_test = x_test[0].squeeze().view(-1, data_dimension)  # 1 x dim
            x_test = Variable(x_test).float()
            z_test = model.mu_z(x_test)

            for n, x_n in enumerate(x_test):
                ax.plot(z_test[n, 0].detach().numpy().flatten(),
                        z_test[n, 1].detach().numpy().flatten(),
                        'o', ms=8, alpha=0.5,
                        color=palette_20_colors[int(labels[n])])

                if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                    z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                    z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                    z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                    z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min, z_0_lim_max])
        plt.ylim([z_1_lim_min, z_1_lim_max])

    plt.savefig(fname=path + filename + '_color.pdf', format='pdf')

def plot_fashionmnist_images(data_loader, model, path, filename='latent_x', max_test_data=1000):
    img_w, img_h = 28, 28
    zoom = 1.3
    fig, ax = plt.subplots(figsize=(16, 16))

    z_0_lim_min, z_0_lim_max = -1.0, 1.0
    z_1_lim_min, z_1_lim_max = -1.0, 1.0

    # Plot point-estimates x or predictive distribution according to type of model
    if isinstance(model, SASGP):

        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.amortization_net(x_test)

                for n, x_n in enumerate(x_test):
                    image = x_n.detach().numpy().reshape((img_w, img_h))
                    im = OffsetImage(image, zoom=zoom, cmap=plt.cm.gray)
                    ab = AnnotationBbox(im, (z_test[n, 0].detach().numpy().flatten(),
                                             z_test[n, 1].flatten().detach().numpy().flatten()),
                                        xycoords='data', frameon=False)
                    ax.add_artist(ab)

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min - 0.1, z_0_lim_max + 0.1])
        plt.ylim([z_1_lim_min - 0.1, z_1_lim_max + 0.1])

    elif isinstance(model, BayesianSASGP):
        for i, x_test in enumerate(data_loader):
            if i < max_test_data:
                x_test = x_test[0].squeeze().view(-1, 784)  # 1 x dim
                x_test = Variable(x_test).float()
                z_test = model.mu_z(x_test)

                for n, x_n in enumerate(x_test):
                    image = x_n.detach().numpy().reshape((img_w, img_h))
                    im = OffsetImage(image, zoom=zoom, cmap=plt.cm.gray)
                    ab = AnnotationBbox(im, (z_test[n, 0].detach().numpy().flatten(),
                                             z_test[n, 1].flatten().detach().numpy().flatten()),
                                        xycoords='data', frameon=False)
                    ax.add_artist(ab)

                    if z_test[n, 0].detach().numpy().flatten() < z_0_lim_min:
                        z_0_lim_min = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 0].detach().numpy().flatten() > z_0_lim_max:
                        z_0_lim_max = z_test[n, 0].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() < z_1_lim_min:
                        z_1_lim_min = z_test[n, 1].detach().numpy().flatten()
                    if z_test[n, 1].detach().numpy().flatten() > z_1_lim_max:
                        z_1_lim_max = z_test[n, 1].detach().numpy().flatten()

        plt.xlim([z_0_lim_min-0.5, z_0_lim_max+0.5])
        plt.ylim([z_1_lim_min-0.5, z_1_lim_max+0.5])

    plt.title(r'SAS -- FashionMNIST (Test data)')
    plt.xlabel(r'$\bm{z}$ -- First dimension')
    plt.ylabel(r'$\bm{z}$ -- Second dimension')
    plt.savefig(fname=path + filename + '_images.pdf', format='pdf', bbox_inches='tight')