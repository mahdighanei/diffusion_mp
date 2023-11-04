from data_generation.dataset import get_train_test_val, TrajectoryDataset
from architechtures.temporal_UNet import TemporalUnet

import os
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
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.hparams['noise_steps'], size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new trajectories....")
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

        x.clamp(-1, 1)
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
    loader_train, _, _ = get_train_test_val(hparams)
    print('data loaded') 
    
    model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=True).to(device)
    diffusion = Diffusion(hparams=hparams, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'])
    mse = nn.MSELoss()
    
    directory, writer = startLog(hparams, model)
    l = len(loader_train)

    for epoch in tqdm(range(hparams['num_epochs'])):
        epoch_running_loss = 0.0
        pbar = tqdm(range(loader_train.__len__()), leave=False)

        for i, (inp, label) in enumerate(loader_train, 0):
            inp = inp.to(device)

            t = diffusion.sample_timesteps(inp.shape[0]).to(device)
            x_t, noise = diffusion.noise_input(inp, t)

            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            epoch_running_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        
        ### evalute & save the model
        writer.add_scalar("MSE_train", epoch_running_loss/(i + 1), global_step=epoch * l + i)
        sampled_trajectories = diffusion.sample(model, n=hparams['batch_size'])
        state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
        torch.save(state, directory + f'model_{epoch}.ckpt')


if __name__ == '__main__':
    trainDiffusion(hparams)