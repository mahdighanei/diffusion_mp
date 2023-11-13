import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from time import sleep

import copy


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

class TrajectoryDataset(Dataset):
    def __init__(self, hparams, mode='train', mp_df=None):
        print(mode)
        self.hparams = hparams
        self.mode = mode
        
        data_folder = hparams['data_folder']
        thresh_plan = hparams['thresh_plan']
        thresh_env =  hparams['thresh_env']
        path = hparams['path']
        file_name = 'MPData'

        # self.makeTrajData()

        mp_data = np.load(hparams['data_folder'] + 'trajdata.npy', allow_pickle=True).item()
        print('\nfile loaded...', mp_data['traj'].shape, mp_data['env_id'].shape)
        self.shape_data = pd.read_json(f'{data_folder}MPObsShapeData{path}.json', orient='index')  # shape info for obs
        self.obs_data = np.load(f'{data_folder}MPObsData{path}.npy').astype(np.float32)  # latent obs embedding

        if mode == 'train':
            cond = (mp_data['plan_id'] < thresh_plan) & (mp_data['env_id'] < thresh_env)
        elif mode == 'test':
            cond = (mp_data['plan_id'] >= thresh_plan) & (mp_data['env_id'] < thresh_env)
        elif mode == 'val':
            cond = (mp_data['env_id'] >= thresh_env)
        
        for k in mp_data.keys():
            mp_data[k] = mp_data[k][cond]
        self.data = mp_data
        self.data_envidx = self.data['env_id']
        self.data_planidx = self.data['plan_id']
        print(self.data['traj'].shape, self.data['env_id'].shape)


    def interpolate_points(self, points, num_samples):
        """
        Linearly interpolate between multiple 3D points to return an array of fixed size.

        :param points: A list of points (x, y, z, ..) of any dim.
        :param num_samples: The fixed size of the output array.
        :return: A numpy array of shape (num_samples, dim) containing interpolated points.
        """

        # Convert the list of points to a numpy array for easier manipulation
        points_before = copy.deepcopy(points)
        points = np.array(points)
        points = np.unique(points, axis=0)
        if not points.shape[0] > 1:
            print('Error')
            print('points_before:', points_before)

        # Calculate the total distance along the path defined by the points
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        total_distance = np.sum(distances)

        # Calculate the distance between each pair of points
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Create an array of evenly spaced distances along the path
        sample_distances = np.linspace(0, total_distance, num_samples)

        # Initialize an array to hold the interpolated points
        interpolated_points = np.zeros((num_samples, points.shape[-1]))

        # print(distances)
        if np.any(distances==0):
            print(distances)
            print(points)
            

        # Interpolate points
        for i in range(1, len(points)):
            # Get the indices of sample points that fall between the current pair of points
            mask = (sample_distances >= cumulative_distances[i-1]) & (sample_distances <= cumulative_distances[i])

            if distances[i-1] == 0:
                pass
            # Calculate the interpolation parameter for these points
            t = (sample_distances[mask] - cumulative_distances[i-1]) / distances[i-1]
            if distances[i-1] == 0:
                print()
                print('t', t, (sample_distances[mask] - cumulative_distances[i-1]), points[i] - points[i-1])

            # Interpolate and assign to the output array
            interpolated_points[mask] = points[i-1] + t[:, np.newaxis] * (points[i] - points[i-1])

        return interpolated_points.astype(np.float32)

    def makeTrajData(self):
        """
        Makes trajectory data from state-action data
        """
        dim = 7

        hparams = self.hparams
        data_folder = hparams['data_folder']
        path = hparams['path']
        file_name = 'MPData'
        mp_data = pd.read_pickle(f'{data_folder}{file_name}{path}.pkl')

        self.data_state = np.array(mp_data['state'].tolist(), dtype=np.float32)[:,(-2*dim):]
        self.data_action = np.array(mp_data['action'].tolist(), dtype=np.float32)
        self.data_envidx = np.array(mp_data['env_id'].tolist())
        self.data_planidx = np.array(mp_data['plan_id'].tolist())
        del mp_data
        print('data state', self.data_state.shape)

        self.data = {k: list() for k in ['traj', 'env_id', 'plan_id', 'goal_state']}
        for envid in tqdm(range(self.data_envidx[-1])):
            env_idx = np.where(self.data_envidx == envid)
            env_idx_first = env_idx[0][0]
            for planid in tqdm(range(self.data_planidx[-1]), desc='plan_id', leave=False):
                idx = env_idx_first + np.where(self.data_planidx[env_idx] == planid)[0]
                states = self.data_state[idx]
                goal_state = states[0][:dim]
                states = states[:, dim:]   # remove goal state
                actions = self.data_action[idx]
                actions[-1] = 0.0  # zero out the last action
                num_points = 64
                interpolated_states = self.interpolate_points(states, num_points)
                interpolated_actions = self.interpolate_points(actions, num_points)
                self.data['traj'].append(np.concatenate((interpolated_states, interpolated_actions), axis=-1,
                                                        dtype=np.float32))
                self.data['env_id'].append(envid)
                self.data['plan_id'].append(planid)
                self.data['goal_state'].append(goal_state)
        for k in self.data.keys():
            self.data[k] = np.array(self.data[k])
        print(self.data['traj'].shape, self.data['env_id'].shape)
        np.save('data/trajdata.npy', self.data, allow_pickle=True)

    def __len__(self):
        return self.data['traj'].shape[0]

    def __getitem__(self, index):
        env_id = self.data['env_id'][index]
        obs_embedding = self.obs_data[env_id][0] # [obs_embedding_dim]
        obs_embedding = torch.from_numpy(obs_embedding).type(torch.float32)

        if self.mode == 'train':
            traj = torch.from_numpy(self.data['traj'][index]).type(torch.float32)   # [horizon * transition_dim]
        else:
            traj = torch.tensor(env_id, dtype=torch.long)

        return traj, obs_embedding

def get_train_test_val(hparams):
    mode = hparams['mode']
    batch_size = hparams['batch_size']
    data_folder = hparams['data_folder']
    path = hparams['path']
    file_name = 'MPData'

    # mp_df = pd.read_pickle(f'{data_folder}{file_name}{path}.pkl')
    mp_df = None
    print('data file loaded')
    # print(mp_df)

    if mode == 'mpnet':     
        dataset = MPNetDataset
    elif mode == 'ae':     
        dataset = AEDataset
    elif mode == 'diffusion':
        dataset = TrajectoryDataset
    seq = ['train', 'test', 'val']
    loader_seq = []
    for it in seq:
        loader = torch.utils.data.DataLoader(dataset(hparams, it, mp_df), batch_size=batch_size,
                                                    shuffle=True, num_workers=10)
        loader_seq.append(loader)
        # if hparams['mode'] == 'diffusion':
        #     return loader
    return loader_seq[0], loader_seq[1], loader_seq[2]