import os
import argparse

import yaml
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import kmedoids
from stable_baselines3.common.utils import set_random_seed

from dynsyn.sb3_runner.wrapper import *  # noqa


class DynSyn:
    def __init__(
        self,
        traj_data: np.ndarray,
        cluster_nums: int,
        cluster_mode: str = "kmedoids",
        logdir: str = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.traj_data = traj_data
        self.verbose = verbose
        self.cluster_mode = cluster_mode
        self.cluster_nums = cluster_nums
        self.logdir = logdir
        self.seed = seed

    def cluster(self, corr) -> list:
        if self.cluster_mode == "kmedoids":
            return self._kmedoids_cluster(corr)
        else:
            raise NotImplementedError

    def _kmedoids_plot(self, dis) -> None:
        """Plot the minimum and maximum distance between the medoids."""

        mean_list, std_list, max_list, min_list = [], [], [], []

        for n_clusters in range(2, dis.shape[0]):
            mean_list.append([])
            std_list.append([])
            max_list.append([])
            min_list.append([])

            for i in range(10):
                km = kmedoids.KMedoids(
                    n_clusters=n_clusters, method="fasterpam", max_iter=10000, random_state=self.seed + i
                )
                c = km.fit(dis)

                dis_ = []
                for i in range(len(c.medoid_indices_)):
                    for j in range(i + 1, len(c.medoid_indices_)):
                        dis_.append(dis[c.medoid_indices_[i], c.medoid_indices_[j]])

                mean_list[-1].append(np.mean(dis_))
                std_list[-1].append(np.std(dis_))
                max_list[-1].append(np.max(dis_))
                min_list[-1].append(np.min(dis_))

            mean_list[-1] = np.mean(mean_list[-1])
            std_list[-1] = np.mean(std_list[-1])
            max_list[-1] = np.mean(max_list[-1])
            min_list[-1] = np.mean(min_list[-1])

        plt.rcParams["font.family"] = "Times New Roman"
        # plt.plot(mean_list, label="mean")
        # plt.plot(std_list, label="std")
        plt.plot(max_list, label="max")
        plt.plot(min_list, label="min")
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.legend()

        plt.axvline(self.cluster_nums, color="k", linestyle="-.")

        plt.xlabel("Number of clusters")
        plt.ylabel("Distance")
        plt.title("The minimum and maximum distance between the medoids")

        plt.savefig(os.path.join(self.logdir, "kmedoids_analysis.png"), dpi=300)

    def _kmedoids_cluster(self, corr) -> list:
        """Cluster the muscles based on the correlation matrix using the KMedoids algorithm."""

        # Convert the correlation matrix to distance matrix
        dis = 1 - corr
        dis = np.where(dis < 0, 0, dis)

        self._kmedoids_plot(dis)

        bins_list = []
        dis_list = []
        for i in range(20):
            km = kmedoids.KMedoids(
                n_clusters=self.cluster_nums, method="fasterpam", max_iter=10000, random_state=self.seed + i
            )
            c = km.fit(dis)

            # Get the bins
            bins = []
            for i in range(self.cluster_nums):
                bins.append(np.where(c.labels_ == i)[0].tolist())
            bins = sorted(bins)
            bins_list.append(bins)
            dis_list.append(c.inertia_)

        # Reture the most frequent bins from bins_list based on the inertia_
        indx_set = {}
        for dis in dis_list:
            if dis not in indx_set:
                indx_set[dis] = 0
            indx_set[dis] += 1
        max_indx = max(indx_set.values())
        max_item = [k for k, v in indx_set.items() if v == max_indx]
        min_dis_max_item = min(max_item)
        return [bins_list[i] for i in range(len(bins_list)) if dis_list[i] == min_dis_max_item][0]

    def calculate_len(self) -> list:
        """Generate the DynSyn representation based on the muscle length data."""

        # Normalize the muscle data for each muscle
        muscle_data = self.traj_data
        min_muscle_data = np.min(muscle_data, axis=0)
        max_muscle_data = np.max(muscle_data, axis=0)
        muscle_data = (muscle_data - min_muscle_data) / (max_muscle_data - min_muscle_data)

        # Calculate the correlation matrix in stages
        group_elements = 100
        assert muscle_data.shape[0] % group_elements == 0
        muscle_data = muscle_data.reshape(
            muscle_data.shape[0] // group_elements, group_elements, muscle_data.shape[1]
        )

        muscle_norm = np.linalg.norm(muscle_data, axis=1)
        muscle_norm = np.expand_dims(muscle_norm, axis=1)
        corr = muscle_data.transpose(0, 2, 1) @ muscle_data
        norm = muscle_norm.transpose(0, 2, 1) @ muscle_norm
        corr /= norm + np.finfo(float).eps
        corr = np.mean(corr, axis=0, keepdims=True).squeeze(0)

        if self.verbose:
            np.savetxt(os.path.join(self.logdir, "corr_matrix.npy"), corr, fmt="%.2f")
            np.save(os.path.join(self.logdir, "corr_matrix_ori.npy"), corr)

        # Cluster the muscles based on the correlation matrix
        bins = self.cluster(corr)

        if self.verbose:
            print(bins)
            print(f"There are {len(bins)} bins")
            print(sum([len(i) for i in bins]))
            print((np.count_nonzero(np.where(corr > 0.7, 1, 0)) - 80) / 2)
            print(corr)
            corr = corr.reshape(-1, 1).squeeze()
            plt.figure()
            plt.hist(corr, 50)
            plt.show()

        return bins


def generate_dynsyn_data_random(args):
    """Generate the muscle length data for the DynSyn algorithm."""

    # Init the environment
    if args.header is not None:
        exec(args.header)

    env = gym.make(args.env_name, **args.env_kwargs if args.env_kwargs is not None else {})

    if args.wrapper is not None:
        for wrapper in args.wrapper.keys():
            env = eval(wrapper)(env, **args.wrapper[wrapper])
    env.reset()

    # Init the buffer
    muscle_len_data_play = np.zeros((args.play_times, args.num_frames, env.model.nu))

    # Init variables
    zero_action = env.action_space.sample() * 0
    assert args.num_frames % args.control_freq == 0

    for i in range(args.play_times):
        env.reset()
        for j in range(args.num_frames // args.control_freq):
            if j % 100 == 0:
                print(f"Play time: {i}/{args.play_times}, frame: {j}/{args.num_frames // args.control_freq}")

            env.data.qvel[:] = np.random.uniform(-1, 1, env.model.nv) * args.control_amp

            for k in range(args.control_freq):
                if args.special_control is not None:
                    exec(args.special_control)

                env.step(zero_action)

                length = env.data.actuator_length.copy()
                muscle_len_data_play[i, j * args.control_freq + k, :] = length

                env.render()

    env.close()

    muscle_len_data_play = muscle_len_data_play.reshape(-1, env.model.nu)

    return muscle_len_data_play


def parse_args():
    """Parse the arguments for the DynSyn algorithm."""
    parser = argparse.ArgumentParser(description="Run the DynSyn algorithm")
    parser.add_argument("-f", type=str)
    parser.add_argument("-e", type=str)
    parser.add_argument("-d", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.f, "r") as f:
        data = yaml.safe_load(f)
        data = data[args.e]
        data["data"] = args.d
        data["seed"] = args.seed

    args = argparse.Namespace(**data)

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)

    args.logdir = os.path.join(args.logdir, args.env_name, str(args.latent_dim), str(args.seed))
    os.makedirs(args.logdir, exist_ok=True)

    if args.data is not None:
        length = np.load(f"{args.data}/muscle_len_data_play.npy")
    else:
        length = generate_dynsyn_data_random(args)
        np.save(f"{args.logdir}/muscle_len_data_play.npy", length)

    dynsyn = DynSyn(length, args.latent_dim, logdir=args.logdir, seed=args.seed)
    bins = dynsyn.calculate_len()

    with open(f"{args.logdir}/bins.txt", "w") as f:
        f.write(str(bins))


if __name__ == "__main__":
    main()
