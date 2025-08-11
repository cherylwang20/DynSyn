import mujoco
import os
import sys
sys.path.append(os.getcwd() + r"\..\myosuite")
sys.path.append(os.getcwd() + r"\..")
#print('\n'.join(sys.path))

from myosuite.utils import gym
import myosuite.envs.myo.myobase


import numpy as np
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
import os
import random
from tqdm.auto import tqdm
import warnings
import argparse
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tostring.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the `reset\(\)` method is not within the observation space.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the `step\(\)` method is not within the observation space.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the `step\\(\\)` method was expecting numpy array dtype to be float32.*")
warnings.filterwarnings("ignore", message=".*The obs returned by the `reset\\(\\)` method was expecting numpy array dtype to be float32.*")

parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--env_name", type=str, default="myofullbodyWalk-v0", help="environment name")
parser.add_argument("--policy", type=str, default='best', help="policy name")
parser.add_argument("--syn", type=bool, default=False, help="whether or not using syn")

args = parser.parse_args()


movie = True
path = os.getcwd()
env_name = args.env_name 


env = gym.make(env_name)

nb_seed = 1

model_num = args.policy 
model = SAC.load('/home/cheryl16/projects/def-durandau/cheryl16/DynSyn/logs/myofullbodyWalk-v0/0709-180733_0/checkpoint/best/model')

m = []

env.reset()

options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# Tweak scales of contact visualization elements
env.unwrapped.sim.model.vis.scale.contactwidth = 0.1
env.unwrapped.sim.model.vis.scale.contactheight = 0.03
env.unwrapped.sim.model.vis.scale.forcewidth = 0.05
env.unwrapped.sim.model.vis.map.force = 0.05

random.seed() 

frames = []
view = 'front'
m_act = []
all_rewards = []

for _ in tqdm(range(1)):
    ep_rewards = []
    done = False
    obs = env.reset()
    step = 0
    muscle_act = []
    acceleration = []
    while (not done) and (step < 200):
          obs = env.unwrapped.obsdict2obsvec(env.unwrapped.obs_dict, env.unwrapped.obs_keys)[1]  
          action, _ = model.predict(obs, deterministic= True)
          obs, reward, done, info, _ = env.step(action)
          ep_rewards.append(reward)
          m.append(action)
          if movie:
                  geom_1_indices = np.where(env.unwrapped.sim.model.geom_group == 1)
                  geom_2_indices = np.where(env.unwrapped.sim.model.geom_group == 2)
                  env.unwrapped.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  env.unwrapped.sim.model.geom_rgba[geom_2_indices, 3] = 0
                  #env.unwrapped.sim.renderer.render_to_window()
                  frame = env.unwrapped.sim.renderer.render_offscreen(width= 640, height=480,camera_id='side_view')
                  frame = (frame).astype(np.uint8)
                  frames.append(frame)

          step += 1
    all_rewards.append(np.sum(ep_rewards))
    m_act.append(muscle_act)

print(all_rewards)

if movie:
    os.makedirs(path+'/videos' +'/' + env_name, exist_ok=True)
    skvideo.io.vwrite(path+'/videos'  +'/' + env_name + '/' + model_num + f'{view}_video.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	