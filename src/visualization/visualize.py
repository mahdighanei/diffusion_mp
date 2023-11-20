from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment
from data_generation.dataset import TrajectoryDataset
from architechtures.temporal_UNet import TemporalUnet
from training.diffusion import Diffusion

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


def plotTraj(env, obs_info, traj):
    """Plot trajectory if done
    """
    # env.stspace.disconnect()

    # stsp = KukaEnv(GUI=True)
    # env = GymEnvironment(stsp)
    # env.populateObstacles(obs_info, removelastIdx=0)
    env.stspace.plot(traj)

def visualizeTrajDataset(hparams):
    """Visualize diffusion from ouptut trajectory file
    """
    data_folder = hparams['data_folder']
    path = hparams['path']
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
        print(traj.shape)
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


def visualizeDiffusionPerformance(mode, hparams):
    """Visualize/Evaluate the Diffusion performance on the dataset
    """
    config = {'use_model': True, 'stochastic': True, 'skip_trivial': False}
    N = 32 #128         # (batch size) Number of actions
    max_tsteps = 30 # max time step for traj

    ### load dataset
    dataset = TrajectoryDataset(hparams=hparams, mode=mode)
    shape_data = dataset.shape_data
    data_env_id = dataset.data_envidx
    data_plan_id = dataset.data_planidx

    ### load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 7

    ### diffusion model
    path_diffusion = 'logs/conditioned_withattention/model_999.ckpt'
    diffusion_model = TemporalUnet(hparams['horizon'], hparams['transition_dim'], attention=True, conditional=True).to(device)
    diffusion_model.load_state_dict(torch.load(path_diffusion)['ema_state_dict'])
    diffusion_model.eval()
    diffusion = Diffusion(hparams, device)

    ### collision model
    path_collision = 'logs/collision/model_ep999.ckpt'
    # collision_model = CollisionModel().to(device)
    # collision_model.load_state_dict(torch.load(path_collision)['state_dict'])
    # collision_model.eval()


    stats = {'num_trivial': 0, 'num_solved': 0, 'num_total': 0}
    print('max_tsteps', max_tsteps)
    goal_bodyid = -1
    start_bodyid = -1
    pre_envid = -1
    pre_planid = -1
    pbar = trange(0, dataset.__len__())
    for i in pbar:
        i = np.random.randint(0, dataset.__len__())  # for random plotting
        envid = data_env_id[i]
        if envid != pre_envid:
            if pre_envid != -1:
                env.stspace.disconnect()
            pre_envid = envid

            stsp = KukaEnv(GUI=True)
            env = GymEnvironment(stsp)
            obs_info = shape_data.iloc[envid]
            env.populateObstacles(obs_info, removelastIdx=0)
        
        info_traj = []
        if data_plan_id[i] != pre_planid:
            pre_planid = data_plan_id[i]
            x, y = dataset.__getitem__(i) # [,H, 2*dim], [, obs_dim]

            env.stspace.init_state = x[0, :dim].cpu().data.numpy().flatten()
            env.stspace.goal_state = x[-1, :dim].cpu().data.numpy().flatten()

            ### plot start and goal
            info_traj = [env.stspace.init_state]
            goal_xyz = env.stspace.get_robot_points(env.stspace.goal_state)
            if goal_bodyid != -1:
                env.stspace.remove_body(goal_bodyid)
            goal_bodyid = env.stspace.add_visual_cube(goal_xyz)
            
            start_xyz = env.stspace.get_robot_points(env.stspace.init_state)
            if goal_bodyid != -1:
                env.stspace.remove_body(start_bodyid)
            start_bodyid = env.stspace.add_visual_cube(start_xyz, 'cube')

            stats['num_total'] += 1
            state = env.reset(sampleStandGoal=False, new_obstacle=False)
            if not env.stspace._state_fp(env.stspace.init_state):
                print('start is in collision')
            if env.stspace._edge_fp(env.state, env.stspace.goal_state):
                stats['num_trivial'] += 1
                if config['skip_trivial']:
                    continue
            
            ### plan using trained model
            for t in range(max_tsteps):
                # input(f't={t}')
                if config['use_model']:
                    input('press enter')
                    with torch.no_grad():
                        obs_embedding = torch.tile(y, (N, 1)).to(device)
                        
                        ###  get traj using diffusion model
                        start = torch.tile(torch.from_numpy(state[dim:]).to(device), (N, 1))
                        goal = torch.tile(x[-1, :dim].to(device), (N, 1))
                        extra_inp = {'start': start, 'goal': goal}
                        # extra_inp = None
                        
                        traj = diffusion.sampleCFG(diffusion_model, N, obs_embedding, extra_inp=extra_inp) # [N, H, 2*dim]
                        print('traj.shape', traj.shape)

                        ## not needed
                        # action = traj[:, 0, dim:]   # get batch of first actions
                        # action = action.cpu().data.numpy().flatten()

                        action = traj[:, 3, :dim] - traj[:, 0, :dim]  # action = state_t3 - state_t0
                        # print(action)
    
                        traj_states = torch.concatenate((traj[0, ::4, :dim], traj[0, -1:, :dim]),
                                                         dim=0).cpu().data.numpy()
                        print(traj_states)
                        print('start', env.stspace.init_state)
                        print('goal', env.stspace.goal_state)
                    
                        plotTraj(env, obs_info, traj_states)

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
                state, r, done, _ = env.step(action)

                
                info_traj.append(state[-dim:])  # for plotting
                
                if r == -1: # if collision detected
                    # input('col*****************\n')
                    if not config['stochastic']:
                        break
                if done:
                    # input('reached goal***************\n')
                    # plotTraj(env, obs_info, info_traj)
                    stats['num_solved'] += 1
                    break
        if i%200 == 0:
            pbar.set_description('envid={} - solved/total: {}/{} ({} trivial)'.format(envid,
                stats['num_solved'], stats['num_total'], stats['num_trivial']))
            