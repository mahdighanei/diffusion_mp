from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment, getPtClouds
from training.trainMPNet import TrainerMPNet
from training.train_diffusion import trainDiffusion

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

    np.set_printoptions(precision=4)
    ######################################################
    ### training
    # trainer_mpnet = TrainerMPNet(hparams)
    # trainer_mpnet.train()

    trainDiffusion(hparams)