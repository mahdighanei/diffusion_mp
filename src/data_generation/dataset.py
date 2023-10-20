import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from time import sleep


class MPNetDataset(Dataset):
    """MPNet data loader
    """
    def __init__(self, hparams, mode='train', mp_df=None):
        print(mode)
        data_folder = hparams['data_folder']
        thresh_plan = hparams['thresh_plan']
        thresh_env =  hparams['thresh_env']
        path = hparams['path']
        self.hparams = hparams
        mp_data = pd.read_json(f'{data_folder}MPData{path}.json', orient='index') if mp_df is None else mp_df
        self.shape_data = pd.read_json(f'{data_folder}MPObsShapeData{path}.json', orient='index')
        obs_path = f'{data_folder}MPObsData{path}.npy'
        print(obs_path)
        self.obs_data = np.load(obs_path).astype(np.float32)

        dim = 7

        if mode == 'train':
            mp_data = mp_data[(mp_data['plan_id'] < thresh_plan) & (mp_data['env_id'] < thresh_env)]
        elif mode == 'test':
            mp_data = mp_data[(mp_data['plan_id'] >= thresh_plan) & (mp_data['env_id'] < thresh_env)]
        elif mode == 'val':
            mp_data = mp_data[mp_data['env_id'] >= thresh_env]
            
        self.data_state = np.array(mp_data['state'].tolist(), dtype=np.float32)[:,(-2*dim):]
        self.data_action = np.array(mp_data['action'].tolist(), dtype=np.float32)
        self.data_envidx = np.array(mp_data['env_id'].tolist())
        self.data_planidx = np.array(mp_data['plan_id'].tolist())
        print('data state', self.data_state.shape, 'shape data', self.shape_data.shape, 'obs data', self.obs_data.shape)

        self.data_next_pos = self.data_state[:,-dim:] + self.data_action

        assert(len(self.data_state) == len(self.data_planidx))

    def __len__(self):
        return self.data_state.shape[0]

    def __getitem__(self, index):
        x = {}
        output = {}

        ### encoding
        max_perenv_idx = np.random.randint(self.obs_data.shape[1])
        if not self.hparams['sample_latent']:
            max_perenv_idx = 0
        obs = self.obs_data[self.data_envidx[index], max_perenv_idx, ...]    # pt-cloud encoding
        x['state'] = np.concatenate((obs, self.data_state[index]))

        ### output
        output['action'] = self.data_action[index, ...]
        
        for k in x.keys():
            x[k] = torch.from_numpy(x[k]).type(torch.float32)
        for k in output.keys():
            output[k] = torch.from_numpy(output[k]).type(torch.float32)

        return x, output

class AEDataset(Dataset):
    """Auto-Encoder data loader
    """
    def __init__(self, hparams, mode='train', trash=None):
        path = hparams['ae_path']
        self.ae_num_ptclouds = hparams['ae_num_ptclouds']
        self.point_cloud = np.load(f'data/obsdata{path}.npy').astype(np.float32)
        print(self.point_cloud.shape)
        
        num_envs = self.point_cloud.shape[0]
        if mode == 'train':
            self.point_cloud = self.point_cloud[:int(.7 * num_envs),...]
        elif mode == 'test':
            self.point_cloud = self.point_cloud[int(0.7 * num_envs) : int(0.85 * num_envs),...]
        elif mode == 'val':
            self.point_cloud = self.point_cloud[int(.85 * num_envs):,...]
    
    def __len__(self):
        return self.point_cloud.shape[0]

    def __getitem__(self, index):
        x = self.point_cloud[index,...]
        idxx = np.random.choice(x.shape[0], self.ae_num_ptclouds, replace=False)  
        x = x[idxx, ...]
        x = np.moveaxis(x, -1, 0)  # get [dim, num_points]
        x = torch.from_numpy(x).type(torch.float32)
        x = {'pt_clouds': x}
        return x, x


def get_train_test_val(hparams):
    mode = hparams['mode']
    batch_size = hparams['batch_size']
    data_folder = hparams['data_folder']
    path = hparams['path']
    file_name = 'MPData' if mode == 'mpnet' else 'MPRobotData'

    mp_df = pd.read_pickle(f'{data_folder}{file_name}{path}.pkl')
    print('data file loaded')
    print(mp_df)

    if mode == 'mpnet':     
        dataset = MPNetDataset
    elif mode == 'ae':     
        dataset = AEDataset
    seq = ['train', 'test', 'val']
    loader_seq = []
    for it in seq:
        loader = torch.utils.data.DataLoader(dataset(hparams, it, mp_df), batch_size=batch_size,
                                                    shuffle=True, num_workers=10)
        loader_seq.append(loader)
    return loader_seq[0], loader_seq[1], loader_seq[2]