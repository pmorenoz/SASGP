
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Pablo Moreno-Munoz, Cilie W. Feldager
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import glob
import os

PATH_TO_FILES = './../data/results/'
# All files and directories ending with .txt and that don't begin with a dot:
args_files_list = glob.glob(PATH_TO_FILES + '*.txt')
for args_file in args_files_list:
    with open(args_file) as f:
        lines = f.readlines()

        filename = lines[2].split(' ')[1].split('\n')[0]
        dataset = lines[3].split(' ')[1].split('\n')[0]
        batch = lines[5].split(' ')[1].split('\n')[0]
        active_set = lines[7].split(' ')[1].split('\n')[0]
        seed = lines[12].split(' ')[1].split('\n')[0]

        newname = 'model_sasgp_' \
                  + dataset \
                  + '_s' + seed \
                  + '_b' + batch \
                  + '_a' + active_set

        os.rename(PATH_TO_FILES + 'args_' + filename + '.txt',
                  PATH_TO_FILES +'args_' + newname + '.txt')

        os.rename(PATH_TO_FILES + filename + '.pt',
                  PATH_TO_FILES +newname + '.pt')

        os.rename(PATH_TO_FILES + 'loss_' + filename + '.pt',
                  PATH_TO_FILES +'loss_' + newname + '.pt')

        os.rename(PATH_TO_FILES + 'time_' + filename + '.pt',
                  PATH_TO_FILES +'time_' + newname + '.pt')