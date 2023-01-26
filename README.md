# SASGP -- Stochastic Active Sets for Gaussian Processes
In this repository, you can find all the files for running the models and experiments described in our paper [Revisiting Active Sets for Gaussian Process Decoders](https://openreview.net/pdf?id=rAVqc7KSGDa). The code aims to train a GP decoder (or GPLVM) in a scalable manner using rediscovered methods from the early days of sparse approximations (i.e. active sets). By running a few experiments, one can quickly perceive the simplicity and easiness for fitting this sort of models, that were notoriously difficult to fit in the past with auxiliary methods widely known in the GP community.

## 🥬 Principal modules and some details -- 

## 🥑 Basic use --

The main file for training the SAS-GP and learn the probabilistic representation is `train.py`. In particular, you can make several combinations among the following choices. 

For extra arguments and options, check `/utils/command_line_parser.py`.

## 🥝 Citation --

Please, if you use this code, include the following citation:
```
@inproceedings{MorenoFeldagerHauberg22,
  title =  {Revisiting Active Sets for {G}aussian Process Decoders},
  author =   {Moreno-Mu\~noz, Pablo and Feldager, Cilie W and Hauberg, S{\o}ren},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year =   {2022}
}
```
