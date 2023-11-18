from data_generation.dataset import get_train_test_val, TrajectoryDataset
from architechtures.temporal_UNet import TemporalUnet
from training.diffusion import Diffusion

import os
import numpy as np
import copy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import datetime
# from utils import *
# from modules import UNet
import time
from torch.utils.tensorboard import SummaryWriter

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def startLog(hparams, model):
    """starts tensorboard logging
    """
    directory = './logs/' + datetime.datetime.now().strftime("%m%d_%H_%M/")
    writer = SummaryWriter(directory)
    with open(directory + 'config.txt', 'w') as f:
        s = "model:\n" 
        for name, layer in model.named_modules():
            s += f'{name}: {str(layer)}\n'

        s += '\n\n**************Hyperparams:\n'
        for k, v in hparams.items():
            s += f'{k}: \t{v}\n' 
        f.write(s)
    return directory, writer

def trainDiffusion(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    ### data loader
    loader_train, _, loader_val = get_train_test_val(hparams)
    print('data loaded') 
    
    if hparams['conditional_DDPM']:
        model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=True, conditional=True)
    else:
        model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=True, conditional=False)
    model = model.to(device)
    diffusion = Diffusion(hparams=hparams, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'])
    loss_fn = nn.MSELoss()
    
    directory, writer = startLog(hparams, model)
    if hparams['use_EMA']:
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in tqdm(range(hparams['num_epochs'])):
        epoch_running_loss = 0.0
        pbar = tqdm(range(loader_train.__len__()), leave=False)

        for i, (inp, label) in enumerate(loader_train, 0):
            inp = inp.to(device)
            label = label.to(device)

            t = diffusion.sample_timesteps(inp.shape[0])
            x_t, noise = diffusion.noise_input(inp, t)

            if hparams['conditional_DDPM']:
                if np.random.random() < 0.1:
                    label = None
                predicted_noise = model(x_t, t, label)
            else:
                predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hparams['use_EMA']:
                ema.step_ema(ema_model, model)

            pbar.update(1)
            epoch_running_loss += loss
        
        ### evaluate & save the model
        writer.add_scalar("MSE_train", epoch_running_loss.item()/(i + 1), global_step=epoch)

        if ((epoch + 1) % hparams['test_interval'] == 0):
            state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            if hparams['use_EMA']:
                state['ema_state_dict'] = ema_model.state_dict()
            torch.save(state, directory + f'model_{epoch}.ckpt')
        
            traj_log = []
            label_log = []
            total_batches = -1
            if hparams['conditional_DDPM']:
                pbar_val = pbar = tqdm(range(loader_val.__len__()), leave=False)
                for i, (inp, label) in enumerate(loader_val, 0):
                    if i >= total_batches:
                        break
                    label = label.to(device)
                    sampled_trajectories = diffusion.sampleCFG(ema_model, hparams['batch_size'], label)
                    traj_log.append(sampled_trajectories.cpu().detach().numpy())
                    label_log.append(label.cpu().detach().numpy())
                    pbar_val.update(1)
                pbar_val.close()
            else:
                for i in range(total_batches):
                    if i >= total_batches:
                        break
                    sampled_trajectories = diffusion.sample(model, hparams['batch_size'])
                    traj_log.append(traj_log.append({'traj': sampled_trajectories.cpu().detach().numpy()}))
            np.save(directory + f'traj_log_ep{epoch}.npy', {'label': np.array(label_log),
                                                            'traj': np.array(traj_log)} )