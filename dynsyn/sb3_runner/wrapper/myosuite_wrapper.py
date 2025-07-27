import gymnasium as gym
from gymnasium.core import Env


class MyosuiteWrapper(gym.Wrapper):
    def __init__(self, env: Env, render_mode="rgb_array"):
        super().__init__(env)

        self.unwrapped.render_mode = render_mode

        # Create alais
        self.model = self.env.sim.model
        self.data = self.env.sim.data

    def render(self, mode=None):
        if self.render_mode == "human":
            self.env.mj_render()
        elif self.render_mode == "rgb_array":
            frame_size = (640, 480)
            return self.env.sim.renderer.render_offscreen(
                frame_size[0], frame_size[1], camera_id=0, device_id=0
            )


class MyosuiteRewardInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.reward_info = {}
        self.last_done = False

    def step(self, action):
        if self.last_done:
            self.reward_info = {}
            self.last_done = False

        obs, reward, done, info = self.env.step(action)

        reward_dict = info["rwd_dict"]
        for key, wt in self.rwd_keys_wt.items():
            current_reward = wt * reward_dict[key]
            if key not in self.reward_info:
                self.reward_info[key] = current_reward
            else:
                self.reward_info[key] += current_reward

    def reset(self, **kwargs):
        self.last_done = True
        return super().reset(**kwargs)
