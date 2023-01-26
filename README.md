# SASGP ~ Stochastic Active Sets for Gaussian Processes
In this repository, you can find all the files for running the models and experiments described in our paper [Revisiting Active Sets for Gaussian Process Decoders](https://openreview.net/pdf?id=rAVqc7KSGDa). The code is design for training a GP decoder (or GPLVM) in a scalable manner using (rediscovered) methods from the early days of sparse approximations (i.e. active sets). By running a few experiments, one can quickly perceive the simplicity and easiness for fitting this sort of models, that were notoriously difficult to fit in the past with auxiliary methods widely known in the GP community.

## 🥑 Basic use 

The main file for training the SAS-GP and learning the probabilistic representation is `train.py`. In particular, you can make several combinations among the following choices. 

`parser.add_argument('--dataset', '-data', default='mnist', choices=['mnist', 'fmnist', 'cifar'], type=str, help='choose dataset')`
`parser.add_argument('--model', '-model', default='sas', choices=['sas', 'baseline'], type=str, help='choose model')`
`parser.add_argument('--inference', '-inf', default='gplvm', choices=['gplvm', 'bayesian'], type=str, help='model to run')`

For extra arguments and options, check `/utils/command_line_parser.py`.

## 🥝 Citation 

Please, if you use this code, include the following citation:
```
@inproceedings{MorenoFeldagerHauberg22,
  title =  {Revisiting Active Sets for {G}aussian Process Decoders},
  author =   {Moreno-Mu\~noz, Pablo and Feldager, Cilie W and Hauberg, S{\o}ren},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year =   {2022}
}
```
