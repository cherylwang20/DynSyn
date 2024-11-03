import gymnasium as gym
import numpy as np


class MuscleNormWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.env.action_space.shape[0],))

    def action(self, action):
        action = 1.0 / (1.0 + np.exp(-5.0 * (action - 0.5)))
        return action
