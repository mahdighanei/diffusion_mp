try:
    from kuka_env import KukaEnv
except:
    pass

# try:
#     from environment.omplsolver import OMPLSolver
# except:
#     # pass
#     from omplsolver import OMPLSolver

import numpy as np
import math
from gym import Env
from shapely.geometry import Point, Polygon
from time import sleep, time

import matplotlib.pyplot as plt


def getPtClouds(stspace):
    """Obtains & processes the point cloud data
    """
    n = 2800

    pts, rgbaImg, segImg = stspace.get_point_cloud(720, 720, stspace.viewMatrix, stspace.projectionMatrix)
    pts_2, rgbaImg_2, segImg_2 = stspace.get_point_cloud(720, 720, stspace.viewMatrix_2, stspace.projectionMatrix)


    # select a subset
    idx = np.random.choice(range(pts.shape[0]), n//2, replace=False)
    ptclouds_1 = pts[idx,:]
    idx = np.random.choice(range(pts_2.shape[0]), n//2, replace=False)
    ptclouds_2 = pts_2[idx,:]


    if False: # for debugging
        figure, axis = plt.subplots(1,2)
        axis[0].imshow(rgbaImg[:,:,:3], interpolation='nearest')
        axis[1].imshow(rgbaImg_2[:,:,:3], interpolation='nearest')
        # depth[depth > 0.99] = 0  # filter out far away objects 
        # axis[1].imshow(depth, cmap='gray', vmin=0, vmax=1)
        # axis[2].imshow(segImg / np.max(segImg), cmap='gray', vmin=0, vmax=1)
        plt.show()

    return np.concatenate((ptclouds_1, ptclouds_2), dtype=np.float32)


def shapeIntersects(obs1, obs2):
    """Checks if two rectangles intersect where 0bs = (halfextents, centerposition)
    """
    x1 = obs1[1][0] - obs1[0][0]
    x2 = obs1[1][0] + obs1[0][0]
    y1 = obs1[1][1] - obs1[0][1]
    y2 = obs1[1][1] + obs1[0][1]
    new_rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    x1 = obs2[1][0] - obs2[0][0]
    x2 = obs2[1][0] + obs2[0][0]
    y1 = obs2[1][1] - obs2[0][1]
    y2 = obs2[1][1] + obs2[0][1]
    rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    return new_rectangle.intersects(rectangle)


class GymEnvironment(Env):
    """Gym enviromnet for 2d experiments
    """
    def __init__(self, statespace, encoder_model = None) -> None:
        super(GymEnvironment, self).__init__()

        self.t_step = 0
        self.num_resets = 0
        self.stspace = statespace
        
        self.obstacle_idx = []
        self.obstacles = []
        self.state_history = []
        self.obs_encoding = None
        self.state = None
        self.omplsolver = None

        self.dim = self.stspace.config_dim 
        self.bound_norm = np.linalg.norm(self.stspace.bound[self.dim:] - self.stspace.bound[:self.dim])

    def __getState(self, internal_state):
        """Returns the full state
        """
        if self.obs_encoding is None:
            return np.concatenate((self.stspace.goal_state , internal_state))
        else:
            return np.concatenate((self.obs_encoding, self.stspace.goal_state , internal_state))

    def __takeAction(self, act, ignore_collision=False):
        """Takes action and updates to new state and returns reward
        """
        reward = None 
        eps = 0.01     # negative reward coefficieint for valid new_state
        
        ### take action
        old_state = self.state
        new_state = np.array(old_state) + act             # continous action
        new_state = np.clip(new_state, self.stspace.bound[:self.dim] + .0001, 
                            self.stspace.bound[self.dim:] - .0001)

        ### calculate reward
        if not self.stspace._edge_fp(old_state, new_state) and not ignore_collision:
            reward = -1.0
        elif self.stspace.in_goal_region(new_state):
            reward = 1.0
            self.state = new_state
        else:
            reward = -eps
            self.state = new_state
        
        self.state_history.append([old_state, self.state])
        return reward, self.state

    def __getObsObsEncoding(self):
        """Get the encoding of the obstacle state
        """
        use_centers = True
        if use_centers:
            encoding = []
            for obs in self.obstacles:
                encoding.extend(np.concatenate((obs[0], obs[1])))
            encoding = np.array(encoding)
        # else:
            # num_samples_per_env = 2800
            # obstacle_samp = []
            # for _ in range(num_samples_per_env):
            #     obstacle_samp.append(self.stspace.sampleRandomObstacle())
            # inp = torch.from_numpy(np.array(obstacle_samp)).type(torch.float32)
            # inp = torch.moveaxis(inp, 0, 1).unsqueeze(0)
            
            # with torch.no_grad():
            #     encoding, d_out = self.encoder_model(inp)
            #     encoding = encoding.cpu().data.numpy().flatten()
                # d_out = d_out.squeeze(0).cpu().data.numpy()
        return encoding
    
    def step(self, act):
        """Takes a step in the env
        """
        info = {}
        self.t_step += 1
        reward, obs = self.__takeAction(act)
        done = self.stspace.in_goal_region(obs)
        state = self.__getState(obs)

        return state, reward, done, info 
    
    def reset(self, sampleStandGoal = True, new_obstacle=True):
        """Resets the GymEnvironment
        """
        self.t_step = 0
        self.num_resets += 1

        ### environment change
        if new_obstacle:
            for idx in self.obstacle_idx:
                self.stspace.remove_body(idx)
            self.obstacle_idx.clear()
            self.populateRandomObstacles(20)
            self.obs_encoding = self.__getObsObsEncoding()
        
        if sampleStandGoal:
            ### sample new start and goal
            tableTop = True
            if tableTop: 
                # try to sample config within the workspace 
                max_config_z = .6 * np.max([it[1][2] + it[0][2] for it in self.obstacles[:-2]]) 
                min_config_y = .1 + np.min([it[1][1] + it[0][1] for it in self.obstacles[:-2]])
                constraint_fn = lambda x : np.logical_and(x > np.array([-.7, min_config_y, .0125]),
                                                          x < np.array([.7, math.inf, max_config_z])).all()
                
                self.stspace.set_random_init_goal(dist=self.bound_norm / 20, dist_worksp=0.1, constraint_fn=constraint_fn)
            else:
                self.stspace.set_random_init_goal(dist=self.bound_norm / 10, dist_worksp=0.5)  # may need clearance
        
        # update current state
        self.state = self.stspace.init_state
        self.stspace.set_config(self.stspace.init_state) # not ncessary
        self.state_history.clear()

        return self.__getState(self.state)
    
    def populateRandomObstacles(self, n):
        """Creates n random obstacles boxes in the environment
        """
        tableTop = True
        self.obstacles.clear()
        self.obstacle_idx.clear()
        lb = 0.3
        hb = 0.7

        if tableTop:
            self._populateRandomObsTableTop(n)
        else:
            for _ in range(n):
                position = np.empty(3)
                position[2] = np.random.uniform(low=-0.1, high=hb+0.15) # higher z
                xy = np.random.uniform(low=-hb, high=hb, size=(2,))
                while np.linalg.norm(xy) < lb or np.linalg.norm(xy) > hb:
                    xy = np.random.uniform(low=-hb, high=hb, size=(2,))
                position[:2] = xy

                half_extents = np.array([.1, .1, .1])
                self.obstacles.append((half_extents, position))
        
        for halfExtents, basePosition in self.obstacles:
            body_idx = self.stspace.create_voxel(halfExtents, basePosition)
            self.obstacle_idx.append(body_idx)
        self.stspace.invisibile_objectIDs = self.obstacle_idx[-2:]
    
    def _populateRandomObsTableTop(self, n):
        """Creates obstacles for table top experiment
        """
        n = 5
        ymin = 0.2
        lb = 0.4
        hb = 0.6
        hb_lim = hb + .1
        i = 0
        while i < n:
            position = np.empty(3)
            position[:2] = np.random.uniform(low=[-hb, ymin], high=[hb, hb_lim])
            while np.linalg.norm(position[:2]) < lb or np.linalg.norm(position[:2]) > hb_lim:
                position[:2] = np.random.uniform(low=[-hb, ymin], high=[hb, hb_lim])
            half_extents = np.random.uniform(low=.025, high=.07, size=(3,))

            new_obs = (half_extents, position)
            intersect = False
            for obs in self.obstacles:
                intersect = shapeIntersects(new_obs, obs)
                if intersect:
                    break
            
            if not intersect:
                half_extents[2] = np.random.uniform(low=0.075, high=0.3)
                position[2] = half_extents[2] + .025 # to make objects sit on table
                self.obstacles.append((half_extents, position))
                i += 1
        
        # prevent `cheating` paths going around
        max_obs_z = np.max([it[1][2] + it[0][2] for it in self.obstacles])  #TODO: have to change visiblity for point-cloud
        self.obstacles.append(([1, 1, .01],[0, 0, max_obs_z+.3]))   # top wall obstacle
        self.obstacles.append(([1, .01, .5],[0, -.3, .5]))     # back wall obstacle          

    def populateObstacles(self, obss, removelastIdx=0):
        """Creates obstacles based on input dictionary
        """
        assert len(self.obstacle_idx) == len(self.obstacles)
        for idx in self.obstacle_idx:
            self.stspace.remove_body(idx)
        self.obstacle_idx.clear()
        self.obstacles.clear()
 
        for i in range(len(obss['x']) - removelastIdx):  # TODO: fix this 
            position = np.array([obss['x'][i], obss['y'][i], obss['z'][i]]) 
            half_extents = np.array([obss['dx'][i], obss['dy'][i], obss['dz'][i]])
            self.obstacles.append((half_extents, position))

            body_idx = self.stspace.create_voxel(half_extents, position)
            self.obstacle_idx.append(body_idx)

        # self.stspace.invisibile_objectIDs = self.obstacle_idx[-2:]

    def calcHeurisitic(self, state, normalized=False):
        dist = self.stspace.distance(state, self.stspace.goal_state)
        if normalized:
            return dist / self.bound_norm
        return dist
    
    def getOptimalActionSeq(self):
        """Obtains the optimal trajectory=(s_i, a_i, r_i, .... r_goal, s_goal) to the goal state
            and updates the current state
        """
        optimal_traj = []
        solution = self.planOMPL(self.state)
        for i in range(1, len(solution)):
            s_t = np.array(solution[i - 1], dtype=np.float32)         # state at t
            s_tp1 = np.array(solution[i], dtype=np.float32)           # state at t+1
            action = s_tp1 - s_t
            r, next_state = self.__takeAction(action, ignore_collision = True)

            done = (i == (len(solution) - 1))
            state_t = self.__getState(s_t)
            state_tp1 = self.__getState(s_tp1)
            
            optimal_traj.append([state_t, state_tp1, action, r, done])
        return optimal_traj

    def setOMPLSovler(self):
        """Sets the ompl motion planner solver
        """
        self.bounds = [self.stspace.bound[:self.dim], self.stspace.bound[self.dim:]]
        self.omplsolver = OMPLSolver(self.stspace._point_in_free_space,
                                     dim=self.dim, bounds_vec=self.bounds, max_action_perdim=0.8, robotsp=self.stspace)
        self.omplsolver.timelimit = 40.0 #15.0 #10.0 #12.0 #10.0

    def planOMPL(self, start):
        """Return a solution (list of list) for the current start/goal
        """
        if self.omplsolver is None:
            raise ValueError('OMPL solver not set')

        self.omplsolver.setStartGoal(start, self.stspace.goal_state)
        solution_list = self.omplsolver.plan()
        solution = []
        for it in solution_list:
            solution.append(np.array(it))
        return solution

if __name__ == "__main__":
    stspace = KukaEnv(GUI=True)

    env = GymEnvironment(stspace)
    # env.setOMPLSovler()

    # states = env.stspace.sample_n_points(2)
    # env.goal = states[1]
    # end_point = env.stspace.get_robot_points()
    # print('\n\nend point',end_point)
    env.reset(new_obstacle=True)


    # print('distance print', np.sum(env.stspace.bound[:env.dim]- env.stspace.bound[env.dim:]))
    # sol = env.planOMPL(env.state)\

    sol = [np.random.random(7) for i in range(5)]
    print('\nsolution len:',len(sol))
    # print(sol[0].shape)
    if sol:
        input('Enter solution')
        gifs = env.stspace.plot(sol, make_gif=False)
        for gif in gifs:
            plt.imshow(gif)
            plt.show()


    # sol = env.getOptimalActionSeq()
    sleep(50)