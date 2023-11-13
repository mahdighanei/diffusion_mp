from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment
from data_generation.dataset import TrajectoryDataset
from architechtures.temporal_UNet import TemporalUnet

import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt




def VisualizeDiffusion(hparams):
    data_folder = hparams['data_folder']
    thresh_plan = hparams['thresh_plan']
    thresh_env =  hparams['thresh_env']
    path = hparams['path']
    file_name = 'MPData'
    dim = 7

    obs_data = np.load(f'{data_folder}MPObsData{path}.npy').astype(np.float32)  # latent obs embedding
    shape_data = pd.read_json(f'{data_folder}MPObsShapeData{path}.json', orient='index')  # shape info for obs
    data = np.load('logs/conditioned_noattention/traj_log_ep949.npy', allow_pickle=True)
    print(data[0]['label'].shape)
    



    for i in range(10):
        b = np.random.randint(0, 142)
        idx = np.random.randint(0, 1024)
        label = data[b]['label'][idx]
        traj = data[b]['traj'][idx, :, :dim]
        np.where(obs_data[:, ...] == label)
        envid = np.where(obs_data[:, ...] == label)[0][0]
        print('level ', label.shape, traj.shape)
        print(envid)
        
        stsp = KukaEnv(GUI=True)
        env = GymEnvironment(stsp)
        obs_info = shape_data.iloc[envid]
        env.populateObstacles(obs_info, removelastIdx=0)

        sol = traj[0::4, :]
        input('Enter solution')
        gifs = env.stspace.plot(sol, make_gif=False)
        for gif in gifs:
            plt.imshow(gif)
            plt.show()
        env.stspace.disconnect()