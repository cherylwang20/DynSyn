import os

import torch as th
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import Video

from dynsyn.sb3_runner.utils import record_video, create_env


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, cyclic_dump_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.cycle_dump_freq = cyclic_dump_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "checkpoint")
        self.best_ckp_path = os.path.join(self.save_path, "best")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.best_ckp_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.check_freq != 0 and self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(os.path.join(self.log_dir, "monitor")), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.best_ckp_path}")

                    self.model.save(os.path.join(self.best_ckp_path, "model.zip"))
                    self.training_env.save(os.path.join(self.best_ckp_path, "env.zip"))
                    if hasattr(self.model, "save_replay_buffer"):
                        self.model.save_replay_buffer(os.path.join(self.best_ckp_path, "replay_buffer.zip"))

        if self.cycle_dump_freq != 0 and self.n_calls % self.cycle_dump_freq == 0:
            ckp_path = os.path.join(self.save_path, f"step_{self.num_timesteps}")
            os.makedirs(ckp_path, exist_ok=True)
            self.model.save(os.path.join(ckp_path, f"model.zip"))
            self.training_env.save(os.path.join(ckp_path, f"env.zip"))
            if hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(os.path.join(ckp_path, f"replay_buffer.zip"))

        return True


class RewardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_freq: int, args, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.args = args

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        # Evalute the env
        env = create_env(
            self.args.env_name, self.args.single_env_kwargs, self.args.wrapper_list, self.args.env_header, 0
        )
        env.reset()
        info = env.step(env.action_space.sample() * 0)[-1]

        total_reward_list = []
        detail_reward_keys = [key for key in info.keys() if key.startswith("reward")]
        detail_reward_list = {key: [] for key in detail_reward_keys}

        for _ in range(1):
            total_reward = 0
            traj_len = 0
            detail_reward = {key: 0 for key in detail_reward_keys}

            obs = env.reset()[0]
            terminated = truncated = False

            while not terminated and not truncated:
                obs = self.training_env.normalize_obs(obs)
                action, _states = self.model.predict(obs)
                all_state = env.step(action)

                info = all_state[-1]
                rewards = all_state[1]
                if len(all_state) == 5:
                    terminated = all_state[2]
                    truncated = all_state[3]
                else:
                    terminated = truncated = all_state[2]

                total_reward += rewards
                detail_reward = {key: detail_reward[key] + info[key] for key in detail_reward_keys}
                traj_len += 1

            total_reward_list.append(total_reward / traj_len)
            for key in detail_reward_keys:
                detail_reward_list[key].append(detail_reward[key] / traj_len)

        env.close()

        # Logging
        self.logger.record("rewards/total_reward", np.mean(total_reward_list))
        for key in detail_reward_keys:
            self.logger.record("rewards/" + key, np.mean(detail_reward_list[key]))

        return True


class VideoRecorderCallback(BaseCallback):
    """
    Custom callback for recording a video and saving it.
    """

    def __init__(
        self, args, record_freq: int, video_dir: str, video_ep_num: int, env_nums: int = 4, verbose=0
    ):
        super(VideoRecorderCallback, self).__init__(verbose)

        self.record_freq = record_freq
        self.video_dir = video_dir
        self.video_ep_num = video_ep_num
        self.args = args
        self.env_nums = env_nums

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq == 0:

            name_prefix = f"{self.args.agent}-{self.num_timesteps}"
            video_frames, video_fps = record_video(
                self.training_env,
                self.model,
                self.args,
                self.video_dir,
                self.video_ep_num,
                name_prefix=name_prefix,
                return_frames=True,
            )

            max_video_idx = 0
            max_video_len = 0
            for idx, frames in enumerate(video_frames):
                if len(frames) > max_video_len:
                    max_video_idx = idx
                    max_video_len = len(frames)

            video_tensor = th.from_numpy(np.array(video_frames[max_video_idx])).unsqueeze(0)
            video_tensor = video_tensor.permute(0, 1, 4, 2, 3)

            self.logger.record(f"eval/video", Video(video_tensor, video_fps), exclude=("stdout"))

        return True


class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.info_keywords = info_keywords
        self.rollout_info = {}

    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}

    def _on_step(self):
        for key in self.info_keywords:
            vals = [info[key] for info in self.locals["infos"]]
            self.rollout_info[key].extend(vals)
        return True

    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record("rollout/" + key, np.mean(self.rollout_info[key]))
