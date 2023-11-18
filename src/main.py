from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment, getPtClouds
from training.trainMPNet import TrainerMPNet
from training.train_diffusion import trainDiffusion
from visualization.visualize import visualizeTrajDataset, visualizeDiffusionPerformance

import torch
import numpy as np
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
from time import sleep, time


import torch.nn as nn
import torch.nn.functional as F


if __name__ == "__main__":
    hparams = yaml.safe_load(Path('./hparams/hparams.yaml').read_text())
    print(hparams)

    # seed = 
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    np.set_printoptions(precision=4)
    ######################################################
    ### training
    # trainer_mpnet = TrainerMPNet(hparams)
    # trainer_mpnet.train()

    trainDiffusion(hparams)

    ######################################################
    ### visualize
    # visualizeTrajDataset(hparams)
    # visualizeDiffusionPerformance(mode='val', hparams=hparams)


    # a = torch.randn((32, 12, 3))
    # b = torch.concatenate((a[0, ::4, ...], a[0, -1:, ...]), dim = 0)
    # print(a[0])
    # print(b)
    # a = np.random.randn(12, 3)
    # print(np.arange(0,12,4))
    # print(a)
    # print(a[::4])
    # print(a[:None:4])