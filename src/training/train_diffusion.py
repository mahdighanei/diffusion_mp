from data_generation.dataset import get_train_test_val, TrajectoryDataset
from architechtures.temporal_UNet import TemporalUnet

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
import logging
import time
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


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

class Diffusion:
    def __init__(self, hparams, device='cuda'):
        self.device = device
        self.hparams = hparams
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.hparams['beta_start'], self.hparams['beta_end'], self.hparams['noise_steps'])
        
    def noise_input(self, x, t):
        """Noises input data

        Args:
            x (torch tensor): uncorropted input data [b, ...]
            t (torch tensor): sampled diffusion timesteps [b, 1]

        Returns:
            torch tensor: noise-corrupted data
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.hparams['noise_steps'], size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.hparams['horizon'], self.hparams['transition_dim'])).to(self.device)
            for i in tqdm(reversed(range(1, self.hparams['noise_steps'])), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()

        # x.clamp(-1, 1)
        return x

    def sampleCFG(self, model, n, labels, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.hparams['horizon'], self.hparams['transition_dim'])).to(self.device)
            for i in tqdm(reversed(range(1, self.hparams['noise_steps'])), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()

        # x.clamp(-1, 1)
        return x

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
        model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=False, conditional=True)
    else:
        model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=True, conditional=False)
    model = model.to(device)
    diffusion = Diffusion(hparams=hparams, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'])
    loss_fn = nn.MSELoss()
    
    directory, writer = startLog(hparams, model)
    l = len(loader_train)
    if hparams['use_EMA']:
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in tqdm(range(hparams['num_epochs'])):
        epoch_running_loss = 0.0
        pbar = tqdm(range(loader_train.__len__()), leave=False)

        for i, (inp, label) in enumerate(loader_train, 0):
            inp = inp.to(device)
            label = label.to(device)

            t = diffusion.sample_timesteps(inp.shape[0]).to(device)
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
            epoch_running_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        
        ### evaluate & save the model
        writer.add_scalar("MSE_train", epoch_running_loss/(i + 1), global_step=epoch * l + i)

        if ((epoch + 1) % hparams['test_interval'] == 0):
            if hparams['conditional_DDPM']:
                traj_log = []
                pbar_val = pbar = tqdm(range(loader_val.__len__()), leave=False)
                for i, (inp, label) in enumerate(loader_val, 0):
                    label = label.to(device)
                    sampled_trajectories = diffusion.sampleCFG(ema_model, hparams['batch_size'], label)
                    traj_log.append({'label': label.cpu().detach().numpy(),
                                      'traj': sampled_trajectories.cpu().detach().numpy()})
                    pbar_val.update(1)
                pbar_val.close()
                np.save(directory + f'traj_log_ep{epoch}.npy', np.array(traj_log))
            else:
                sampled_trajectories = diffusion.sample(model, hparams['batch_size'])
            state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            if hparams['use_EMA']:
                state['ema_state_dict'] = ema_model.state_dict()
            torch.save(state, directory + f'model_{epoch}.ckpt')