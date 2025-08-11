from typing import List

import gymnasium as gym
from gymnasium.wrappers import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder

from dynsyn.sb3_runner.wrapper import *


def create_env(
    env_name: str,
    single_env_kwargs: dict,
    wrapper_list: dict,
    env_header: str = None,
    seed: int = 0,
    render_mode: str = "rgb_array",
):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    if env_header:
        exec(env_header, globals())

    set_random_seed(seed)
    env = gym.make(env_name, render_mode=render_mode, **single_env_kwargs)
    env.unwrapped._camera_id = 2

    for wrapper_name, wrapper_args in wrapper_list.items():
        try:
            env = eval(wrapper_name)(env, **wrapper_args)
        except NameError:
            print(f"Wrapper {wrapper_name} not found!")
            raise NameError

    return env


def create_vec_env(
    env_name,
    single_env_kwargs,
    env_nums,
    env_header: str = None,
    wrapper_list: dict = None,
    monitor_dir: str = None,
    monitor_kwargs: str = None,
    seed: int = 0,
):
    if hasattr(monitor_kwargs, "info_keywords"):
        monitor_kwargs["info_keywords"] = tuple(monitor_kwargs["info_keywords"])

    vec_env = make_vec_env(
        create_env,
        env_kwargs={
            "env_header": env_header,
            "env_name": env_name,
            "single_env_kwargs": single_env_kwargs,
            "wrapper_list": wrapper_list,
            "seed": seed,
        },
        n_envs=env_nums,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
    )

    return vec_env


def record_video(
    vec_norm,
    model,
    args,
    video_dir: str,
    video_ep_num: int,
    name_prefix: str = "video",
    return_frames: bool = False,
):
    env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        1,
        env_header=args.env_header,
        wrapper_list=args.wrapper_list,
        monitor_dir=None,
        monitor_kwargs=None,
        seed=args.seed,
    )

    try:
        base_env = env.envs[0]
        if hasattr(base_env, "sim") and hasattr(base_env.sim.model, "cam_select"):
            base_env.sim.model.cam_select = camera_id
    except Exception as e:
        print(f"[Warning] Could not set custom camera: {e}")

    vec_env = VecVideoRecorder(
        env,
        video_folder=video_dir,
        record_video_trigger=lambda x: x == 0,
        video_length=10000,
        name_prefix=name_prefix,
    )

    video_frames = []

    for _ in range(video_ep_num):
        done = False
        total_reward = 0

        obs = vec_env.reset()
        while not done:
            obs = vec_norm.normalize_obs(obs)
            action, _ = model.predict(obs, deterministic=False)
            obs, r, done, info = vec_env.step(action)
            total_reward += r

        video_frames.append(vec_env.video_recorder.recorded_frames)
        video_fps = vec_env.video_recorder.frames_per_sec
        print(f"Episode reward: {total_reward}")

    vec_env.close()

    if return_frames:
        return video_frames, video_fps
