from architechtures.AE_model import AENet3
from architechtures.agent_model import MPNet1, MPNet2, c2gHOF
from architechtures.selfsupervised_model import SelfSupervisedModel
from data_generation.dataset import MPNetDataset, get_train_test_val
from utilities.utils import ChamferDistance

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

from time import sleep
from tqdm import tqdm, trange
import os
import copy
import time
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


###################################################################
###### Train the MPNet model
###################################################################
def lossCriterionMPNet(pred, target):
    """the loss fucnction for MPNet
    """
    loss = {}
    loss_val = 0
    key = 'action' #'c2g'
    
    loss['loss_' + key] = F.l1_loss(pred[key], target[key])
    loss_val += loss['loss_' + key]
    
    loss['loss'] = loss_val
    return loss

class lossCriterionAE():
    def __init__(self) -> None:
        self.recons_lossFn = ChamferDistance()

    def __call__(self, pred, target):
        return {'loss': self.recons_lossFn(pred, target['pt_clouds'])}
    

class TrainerMPNet(object):
    """Training class for any model
    """
    def __init__(self, hparams) -> None:
        self.hparams = hparams
        self.mode = hparams['mode']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.log_subfolder = None
        self.startepoch = 0
        self.loss_keys = ['loss']
        ### loss function & network
        if self.mode == 'mpnet':
            dim = 7
            self.loss_keys.extend(['loss_action'])
            self.criterion = lossCriterionMPNet
            self.net = MPNet1(state_dim=hparams['ae_latentsize'] + 2*dim, action_dim=dim)
        elif self.mode == 'ae':
            self.criterion = lossCriterionAE()
            self.net = AENet3(hparams)
        print(f'created model - epoch: {self.startepoch}' )
        self.net = self.net.to(self.device)

        ### data loader
        self.loader_train, self.loader_test, self.loader_val = get_train_test_val(hparams)
        print('data loaded')
        
        ### optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
        self.writer = None
    
    def startLog(self):
        """starts tensorboard logging
        """
        if self.log_subfolder is None:
            self.log_subfolder =  datetime.datetime.now().strftime("%m%d_%H_%M/")
        self.directory = './logs/' + self.log_subfolder
        self.writer = SummaryWriter(self.directory)
        with open(self.directory + 'config.txt', 'w') as f:
            s = "model:\n" 
            for name, layer in self.net.named_modules():
                s += f'{name}: {str(layer)}\n'

            s += '\n\n**************Hyperparams:\n'
            for k, v in self.hparams.items():
                s += f'{k}: \t{v}\n' 
            f.write(s)

    def train(self):
        """Start training, validate at step and log the model/progress
        """
        self.test(self.startepoch, mode='test')
        self.test(self.startepoch, mode='val')
        with tqdm(total=self.hparams['num_epochs']) as pbar:
            for epoch in range(self.startepoch, self.hparams['num_epochs']):
                epoch_running_loss = dict.fromkeys(self.loss_keys, 0.0)
                with tqdm(total=len(self.loader_train), desc='batch', leave=False) as pbar2:
                    for i, data in enumerate(self.loader_train, 0):
                        x, y = data
                        x = {k: v.to(self.device) for k, v in x.items()}
                        y = {k: v.to(self.device) for k, v in y.items()}
                        self.optimizer.zero_grad()

                        ### forward + backward + optimize
                        outputs = self.net(x)
                        if self.mode == 'ae':
                            encoding, outputs = outputs[0], outputs[1]
                        loss = self.criterion(outputs, y)

                        loss['loss'].backward()
                        self.optimizer.step()

                        # print statistics
                        for _, (k, val) in enumerate(epoch_running_loss.items()):
                            epoch_running_loss[k] += loss[k].item()
                        pbar2.update(1)
                        
                for _, (k, val) in enumerate(epoch_running_loss.items()):
                    self.writer.add_scalar('Loss/train_epoch_' + k, val / (i + 1), epoch)

                if (epoch + 1) % self.hparams['test_interval'] == 0:
                    self.test(epoch, mode='test')
                    self.test(epoch, mode='val')
                    state = {
                        'epoch': epoch,
                        'state_dict': self.net.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'loss': epoch_running_loss['loss'] / (i + 1)
                    }
                    torch.save(state, self.directory + f'model_{epoch}.ckpt')
                pbar.set_description('Loss {:.3f}'.format(epoch_running_loss['loss'] / (i + 1)))
                pbar.update(1)
    
    def test(self, epoch, mode='test'):
        """validates the model
        """
        if self.mode == 'selfsupervised' and mode == 'test':
            return
        net = copy.deepcopy(self.net)
        net.eval()
        if mode == 'test':
            data_loader = self.loader_test
        elif mode == 'val':
            data_loader = self.loader_val
        running_loss = dict.fromkeys(self.loss_keys, 0.0)
        with torch.no_grad():
            with tqdm(total=len(self.loader_train), desc='batch', leave=False) as pbar2:
                for i, data in enumerate(data_loader, 0):
                    x, y = data
                    x = {k: v.to(self.device) for k, v in x.items()}
                    y = {k: v.to(self.device) for k, v in y.items()}

                    outputs = net(x)
                    if self.mode == 'ae':
                        encoding, outputs = outputs[0], outputs[1]
                    loss = self.criterion(outputs, y)

                    # print statistics
                    for _, (k, val) in enumerate(running_loss.items()):
                        running_loss[k] += loss[k].item()
                    pbar2.update(1)
                if self.writer is None:
                    self.startLog()
                for _, (k, val) in enumerate(running_loss.items()):
                    self.writer.add_scalar('Loss/' + mode + '_epoch_' + k, val / (i + 1), global_step=epoch)
                