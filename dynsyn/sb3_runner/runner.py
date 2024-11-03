import os
import time
import argparse
import json

import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from dynsyn.sb3_runner.callback import (
    SaveOnBestTrainingRewardCallback,
    VideoRecorderCallback,
    RewardCallback,
    TensorboardCallback,
)
from dynsyn.sb3_runner.utils import create_vec_env, record_video, create_env
from dynsyn.algorithms import *  # noqa


def load_policy(args, Agent):
    if args.env_header:
        exec(args.env_header)

    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy

    if policy not in Agent.policy_aliases.keys():
        policy = eval(policy)

    return policy


def load_agent(args):
    # Get the agent
    if hasattr(sb3, args.agent) or hasattr(sb3_contrib, args.agent):
        # Use the sb3 agent
        Agent = getattr(sb3_contrib, args.agent, getattr(sb3, args.agent, None))
    else:
        # Use the custom agent
        if args.env_header:
            exec(args.env_header)

        Agent = eval(args.agent)

    return Agent


def register_callback(args, video_dir, log_dir):
    # Callback
    callback_list = []
    # Convert to total steps
    args.check_freq //= args.env_nums
    args.record_freq //= args.env_nums
    args.dump_freq //= args.env_nums
    args.reward_freq //= args.env_nums

    if args.reward_freq > 0:
        callback_list.append(RewardCallback(args.reward_freq, args))
    if args.check_freq > 0 or args.dump_freq > 0:
        checkpoint_callback = SaveOnBestTrainingRewardCallback(
            check_freq=args.check_freq, cyclic_dump_freq=args.dump_freq, log_dir=log_dir, verbose=1
        )
        callback_list.append(checkpoint_callback)
    if args.record_freq > 0:
        video_callback = VideoRecorderCallback(
            args, args.record_freq, video_dir=video_dir, video_ep_num=5, verbose=1
        )
        callback_list.append(video_callback)

    callback_list.append(TensorboardCallback(getattr(args, "info_keywords", {})))

    return callback_list


def init_wandb(args, run_name: str) -> None:
    if not hasattr(args, "wandb") or not args.wandb.get("use_wandb", False):
        return

    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
        ) from e

    wandb.init(
        project=args.wandb["project_name"],
        config=vars(args),
        sync_tensorboard=True,
        save_code=False,
        name=run_name,
        notes=args.wandb.get("notes", ""),
    )


def build_env(args, monitor_dir):
    # Parallel environments
    vec_env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        args.env_nums,
        env_header=args.env_header,
        wrapper_list=args.wrapper_list,
        monitor_dir=monitor_dir,
        monitor_kwargs=getattr(args, "monitor_kwargs", {}),
        seed=args.seed,
    )

    # Vec Norm
    if args.vec_normalize["is_norm"] and not args.load_model_dir:
        vec_env = VecNormalize(vec_env, **args.vec_normalize["kwargs"])

    return vec_env


def process_variable(args):
    if args.env_header:
        exec(args.env_header)

    # Processing the learning rate
    if "learning_rate" in args.agent_kwargs and not isinstance(args.agent_kwargs["learning_rate"], float):
        if args.agent_kwargs["learning_rate"].startswith("linear_schedule"):
            from dynsyn.sb3_runner.schedule import linear_schedule
        args.agent_kwargs["learning_rate"] = eval(args.agent_kwargs["learning_rate"])

    # Agent seed
    args.agent_kwargs["seed"] = args.seed

    # Policy kwargs
    if "policy_kwargs" in args.agent_kwargs:
        if "features_extractor_class" in args.agent_kwargs["policy_kwargs"]:
            args.agent_kwargs["policy_kwargs"]["features_extractor_class"] = eval(
                args.agent_kwargs["policy_kwargs"]["features_extractor_class"]
            )

    return args


def train(args):
    # Log dir
    env_name_log = args.env_name.split("/")[-1]
    run_name = os.path.join(env_name_log, time.strftime("%m%d-%H%M%S") + "_" + str(args.seed))
    log_dir = os.path.join(args.log_root_dir, run_name)
    monitor_dir = os.path.join(log_dir, "monitor")
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    video_dir = os.path.join(log_dir, "video")

    if not args.play:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(monitor_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        json.dump(args.total_config, open(os.path.join(log_dir, "config.json"), "w"), indent=4)

    # Callback
    callback_list = register_callback(args, video_dir, log_dir)

    # Model and agent
    Agent = load_agent(args=args)

    if args.play:
        assert args.load_model_dir is not None, "Please specify the model to load"
        print(f"Loading model from {args.load_model_dir}")
        model = Agent.load(
            os.path.join(args.load_model_dir, "model.zip"),
            **args.load_kwargs if hasattr(args, "load_kwargs") else {},
        )

        evaluate(model, args, is_record=True, render_mode="rgb_array")
        return

    # Init wandb
    init_wandb(args, run_name)

    # If not play, train the model
    # Process config variable
    args = process_variable(args)

    vec_env = build_env(args, monitor_dir)

    if args.load_model_dir:
        print(f"Loading model from {args.load_model_dir}")
        vec_env = VecNormalize.load(os.path.join(args.load_model_dir, "env.zip"), vec_env)
        model = Agent.load(
            os.path.join(args.load_model_dir, "model.zip"),
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            **args.agent_kwargs,
        )
        if args.load_buffer:
            model.load_replay_buffer(os.path.join(args.load_model_dir, "replay_buffer.zip"))
    else:
        policy = load_policy(args, Agent)
        model = Agent(policy, env=vec_env, verbose=1, tensorboard_log=log_dir, **args.agent_kwargs)

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback_list)

    # Save the final model
    final_ckp_path = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_ckp_path, exist_ok=True)
    model.save(os.path.join(final_ckp_path, "model.zip"))
    vec_env.save(os.path.join(final_ckp_path, "env.zip"))
    if hasattr(model, "save_replay_buffer"):
        model.save_replay_buffer(os.path.join(final_ckp_path, "replay_buffer.zip"))


def evaluate(model, args, is_record=False, render_mode="human"):
    env = create_env(
        args.env_name, args.single_env_kwargs, args.wrapper_list, args.env_header, 0, render_mode
    )
    vec_norm = VecNormalize.load(os.path.join(args.load_model_dir, "env.zip"), DummyVecEnv([lambda: env]))

    if is_record:
        record_video(
            vec_norm,
            model,
            args,
            video_dir="./output_video",
            video_ep_num=5,
            name_prefix=args.env_name.split("/")[-1],
        )
    else:
        for i in range(100):
            print(i)
            obs = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                obs = vec_norm.normalize_obs(obs)
                action, _states = model.predict(obs)
                obs, rewards, terminated, truncated, info = env.step(action)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-f", type=str, default=None, help="Config file path")

    # The following arguments will overwrite the config file
    parser.add_argument("--wandb_notes", "-m", type=str, default=None, help="Wandb notes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    arg_config = argparse.Namespace(**config)
    arg_config.total_config = config

    if args.wandb_notes is not None:
        arg_config.wandb["notes"] = args.wandb_notes
    if args.seed is not None:
        arg_config.seed = args.seed

    return arg_config


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
