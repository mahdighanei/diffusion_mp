import numpy as np
import torch
from tqdm import tqdm


class Diffusion:
    def __init__(self, hparams, device='cuda'):
        self.device = device
        self.hparams = hparams
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.hparams['beta_start'], self.hparams['beta_end'], self.hparams['noise_steps'])
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=torch.float32)
        
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
        return torch.randint(low=1, high=self.hparams['noise_steps'], size=(n,), device=self.device)

    def sample(self, model, n, extra_inp=None):
        dim = self.hparams['dim']
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.hparams['horizon'], self.hparams['transition_dim']), device=self.device)
            if extra_inp is not None:
                x[:, 0, :dim] = extra_inp['start']
                x[:, -1, :dim] = extra_inp['goal']
                x[:, -1, dim:] = 0.0  # last action zero
            for i in tqdm(reversed(range(1, self.hparams['noise_steps'])), position=0, leave=False):
                t = (torch.ones(n, device=self.device) * i).long()
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                if extra_inp is not None:
                    x[:, 0, :dim] = extra_inp['start']
                    x[:, -1, :dim] = extra_inp['goal']
                    x[:, -1, dim:] = 0.0  # last action zero
        model.train()

        # x.clamp(-1, 1)
        return x

    def sampleCFG(self, model, n, labels, cfg_scale=3, extra_inp=None):
        dim = self.hparams['dim']
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.hparams['horizon'], self.hparams['transition_dim']), device=self.device)
            if extra_inp is not None:
                x[:, 0, :dim] = extra_inp['start']
                x[:, -1, :dim] = extra_inp['goal']
                x[:, -1, dim:] = 0.0  # last action zero
            for i in tqdm(reversed(range(1, self.hparams['noise_steps'])), position=0, leave=False):
                t = (torch.ones(n, device=self.device) * i).long()
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

                if extra_inp is not None:
                    x[:, 0, :dim] = extra_inp['start']
                    x[:, -1, :dim] = extra_inp['goal']
                    x[:, -1, dim:] = 0.0  # last action zero
        model.train()

        # x.clamp(-1, 1)
        return x
