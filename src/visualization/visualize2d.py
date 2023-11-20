from data_generation.dataset import TrajectoryDataset
from architechtures.temporal_UNet import TemporalUnet
from training.diffusion import Diffusion

import numpy as np
import random
import os
import copy
import torch
import pandas as pd
from tqdm import tqdm, trange
import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse



class RectangleObs(object):
    def __init__(self, center, x_len, y_len):
        super(RectangleObs, self).__init__()
        self.name = 'rect'
        self.center = center

        self.x_len = x_len
        self.y_len = y_len

    def isPointCollision(self, point, obsinfl_factor=0.02):
        x_len = (1 + obsinfl_factor) * self.x_len
        y_len = (1 + obsinfl_factor) * self.y_len
        x_range = (self.center[0] - x_len / 2.0, self.center[0] + x_len / 2.0)
        y_range = (self.center[1] - y_len / 2.0, self.center[1] + y_len / 2.0)

        if ((x_range[0] <= point[0] <= x_range[1]) and 
                (y_range[0] <= point[1] <= y_range[1])):
            return True
        return False


class StateSpace2D(object):
    """State-space for 2d obstacle environment.
    """
    def __init__(self) -> None:
        super(StateSpace2D, self).__init__()
        self.obstacles = []
        self.bounds = [0, 1]
    
    def sampleRandom(self):
        """Sample a random state
        """
        a = self.bounds[0]
        b = self.bounds[1]
        return [random.uniform(a, b), random.uniform(a, b)]

    def sampleRandomFree(self):
        """Samples a free random state
        """
        randsample = self.sampleRandom()
        while self.isCollision(randsample):
            randsample = self.sampleRandom()
        return randsample

    def sampleRandomObstacle(self):
        randsample = self.sampleRandom()
        while not self.isCollision(randsample):
            randsample = self.sampleRandom()
        return randsample

    def isCollision(self, point):
        """Checks whether a point is in collision
        """
        pt = np.array(point)
        if np.any((pt < self.bounds[0]) | (pt > self.bounds[1])):         
            return True
        
        for obs in self.obstacles:
            if obs.isPointCollision(point):
                return True
        return False
    
    def isLineCollision(self, point1, point2, resolution=10, use_stspace_extent=False):
        """Checks wether a line  is in collision
        """        
        if use_stspace_extent:
            global_res = 0.015
            stsp_extent = self.bounds[1] - self.bounds[0]
            resolution = int(np.linalg.norm(point2 - point1) / (global_res * stsp_extent))
        resolution += 1
        
        pt1 = np.array(point1)
        pt2 = np.array(point2)
        if np.any((pt1 < self.bounds[0]) | (pt1 > self.bounds[1])) or \
            np.any((pt2 < self.bounds[0]) | (pt2 > self.bounds[1])):         
            return True

        x = point1[0]
        y = point1[1]
        dx = (point2[0] - x) / resolution
        dy = (point2[1] - y) / resolution
        for i in range(resolution + 1):
            if self.isCollision([x + i * dx, y + i * dy]):
                return True
        return False


    def unNormalize(self, inp, stats_key, data_stats):
        return inp * data_stats[stats_key]['std'] + data_stats[stats_key]['mean']
    
    def normalize(self, inp, stats_key, data_stats):
        return (inp -  data_stats[stats_key]['mean']) / data_stats[stats_key]['std']

    def visualizeMPNet(self, hparams):
        """Plots the environment with the obstacles
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {'use_model': True, 'stochastic': True, 'skip_trivial': False}
        N = 6 #32 #128         # (batch size) Number of actions
        max_tsteps = 1 #30 # max time step for traj

        dim = 2
        ### load dataset
        mode = 'val'
        dataset = TrajectoryDataset(hparams=hparams, mode=mode)
        shape_data = dataset.shape_data
        data_env_id = dataset.data_envidx
        data_plan_id = dataset.data_planidx

        ### diffusion model
        path_diffusion = 'logs/model2d/model_999.ckpt'
        diffusion_model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=True, conditional=True).to(device)
        diffusion_model.load_state_dict(torch.load(path_diffusion)['ema_state_dict'])
        diffusion_model.eval()
        diffusion = Diffusion(hparams, device)

        data_stats = None
        # data_stats = getDataStats(mp_df, obs_df)

        stats = {'num_solved': 0, 'num_trivial': 0, 'num_total': 0}

        pre_envid = -1
        pre_planid = -1
        pbar = trange(0, dataset.__len__())
        for i in pbar:
            i = random.randint(0,  dataset.__len__())  # for random selection
            print('data', i)
            envid = dataset.data['env_id'][i]
            if envid != pre_envid:
                pre_envid = envid
                self.env_id = envid
                obs = dataset.shape_data.iloc[envid]
                # env.populateObstacles(obs)
                self.obstacles = []
                for j in range(len(obs['center_x'])):
                    center = [obs['center_x'][j], obs['center_y'][j]]
                    self.obstacles.append(RectangleObs(center, obs['x_len'][j], obs['y_len'][j]))
            
            if data_plan_id[i] != pre_planid:
                pre_planid = data_plan_id[i]
                self.plan_id = data_plan_id[i]
                x, y = dataset.__getitem__(i)

                ### set start and goal
                stats['num_total'] += 1
                self.goal = x[-1, :dim].cpu().data.numpy().flatten()
                self.start = x[0, :dim].cpu().data.numpy().flatten()
                state = copy.deepcopy(self.start)
                # print('start/goal', self.start, self.goal)
                
                if not self.isLineCollision(self.start, self.goal, use_stspace_extent=True): # if collision detected
                    stats['num_trivial'] += 1
                
                traj_plot = []
                ### plan using trained model or visualize dataset
                for t in range(max_tsteps):
                    # input('press enter')
                    if config['use_model']:
                        with torch.no_grad():
                            obs_embedding = torch.tile(y, (N, 1)).to(device)
                            
                            ###  get traj using diffusion model
                            start = torch.tile(torch.from_numpy(state).to(device), (N, 1))
                            goal = torch.tile(x[-1, :dim].to(device), (N, 1))
                            extra_inp = {'start': start, 'goal': goal}
                            # extra_inp = None
                            
                            traj = diffusion.sampleCFG(diffusion_model, N, obs_embedding, extra_inp=extra_inp) # [N, H, 2*dim]
                            # traj = x.unsqueeze(0) # GT data
                            # traj = torch.randn((N, 64, 2*dim), device=device)
                            # print(8'traj.shape', traj.shape)

                            ## not needed
                            # action = traj[:, 0, dim:]   # get batch of first actions
                            # action = action.cpu().data.numpy().flatten()

                            action = traj[:, 3, :dim] - traj[:, 0, :dim]  # action = state_t3 - state_t0
                            # print(action)
        
                            traj_states = torch.concatenate((traj[0, ::8, :dim], traj[0, -1:, :dim]),
                                                            dim=0).cpu().data.numpy()
                            print(traj_states)

                            self.plotScene(traj_states)

                            ### Collision model - choose action based on collision prob
                            inp = {'obs_embedding': obs_embedding, 'state': traj[:, 0, ...]}
                            print('inp', inp['obs_embedding'].shape, inp['state'].shape)

                            # col_prob = collision_model(inp)
                            # col_prob = F.softmax(col_prob, dim=-1).cpu().data.numpy()[:, 1]
                            # best_act_idx = np.argmin(col_prob, axis=-1)
                            best_act_idx = 0
                            # action = traj[best_act_idx, 0, dim:].cpu().data.numpy()
                            action = action[best_act_idx].cpu().data.numpy()
                    else:
                        action = x[t, dim:].cpu().data.numpy().flatten()
                    
                    if False:
                        # state, r, done, _ = env.step(action)
                        done = np.linalg.norm(state - self.goal) < 0.15 #0.25

                        ### log data
                        next_state = state + action
                        traj_plot.append((state, next_state))

                        if not self.config['use_MPNet'] and done:
                            del traj_plot[-1]
                        
                        if self.isLineCollision(state, next_state,  use_stspace_extent=True): # if collision detected
                            self.plotScene(traj_plot)
                            if not self.config['stochastic']:
                                break
                        else:
                            state = next_state
                        
                        if done:
                            stats['num_solved'] += 1
                            self.plotScene(traj_plot)
                            break
        
            if i%1000 == 0:
                pbar.set_description('solved/total: {}/{} ({} trivial)'.format(stats['num_solved'],
                stats['num_total'], stats['num_trivial']))


    def plotScene(self, traj):
        fig, ax = plt.subplots()
        
        # plot start and goal
        sz = 0.02
        if self.start is not None and self.goal is not None:
            ax.add_patch(Circle(self.start, sz, fill=True, color='c', label='start'))
            goal_rec = (self.goal[0] - sz, self.goal[1] - sz)
            ax.add_patch(Rectangle(goal_rec, 2*sz, 2*sz, fill=True, color='g', label='goal'))

        # plot states
        if traj is not None:
            for i in range(len(traj) - 1):
                state = traj[i]
                next_state = traj[i + 1]
                x = [state[0], next_state[0]]
                y = [state[1], next_state[1]]
                ax.plot(x,  y, color='r')
                if i != 0:
                    ax.scatter(state[0], state[1], color='r')

        # plot obstacles
        for obs in self.obstacles:
            obs_shape = None
            if isinstance(obs, RectangleObs):  
                x = obs.center[0] - obs.x_len / 2.0
                y = obs.center[1] - obs.y_len / 2.0
                obs_shape = Rectangle((x,y), obs.x_len, obs.y_len, fill=True, color='k')
                print(np.array([obs.center[0], obs.center[1], obs.x_len, obs.y_len]))
            ax.add_patch(obs_shape)
        plt.legend()
        plt.xlim(self.bounds)
        plt.ylim(self.bounds)
        print('saving plot', i)
        name = 'mpnet' if  self.config['use_MPNet'] else 'gt'
        plt.savefig(f'data/pics/Env{self.env_id}_{self.plan_id}_{name}.png', bbox_inches='tight')
        plt.show()
        plt.close()