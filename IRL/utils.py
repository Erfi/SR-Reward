import os
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Sequence
import logging
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv
from mani_skill2.envs import pick_and_place
from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout
from imitation.data.huggingface_utils import (
    trajectories_to_dataset,
    TrajectoryDatasetSequence,
)
from IRL.replay_buffers import ReplayBufferSamples, ReplayBufferSamplesPlus
from tqdm import tqdm
import h5py


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def add_hist(states, h):
    """
    Takes in a numpy array of size (n, dim) and returns a numpy array of size (n - h + 1, h, dim)
    where each element is a history of h elements.
    e.g. [[1,2], [3,4], [5,6], [7,8]] with h==2 -> [[[1,2], [3,4]], [[3,4], [5,6]], [[5,6], [7,8]]]
    """
    return np.stack([states[i : i + h] for i in range(len(states) - h + 1)])


def evaluate_reward(rollouts, visitnet, h=1, noise=0):
    """
    Evaluate the reward function on the rollouts.
    Args:
        rollouts: (Sequence) Rollouts.
        visitnet: (nn.Module) visitnet for canculating SR.
    Returns:
        (np.ndarray) Rewards.
    """
    returns = []
    for rollout in rollouts:
        obs = add_hist(rollout.obs[:-1], h=h)
        acts = rollout.acts[h - 1 :]
        obs = torch.from_numpy(obs).float().to(visitnet.device)
        acts = torch.from_numpy(acts).float().to(visitnet.device)
        if noise > 0:
            obs += torch.randn_like(obs) * noise
            acts += torch.randn_like(acts) * noise
        with torch.no_grad():
            sr = visitnet(obs, acts)
        rewards = sr_to_reward(sr)
        returns.append(torch.sum(rewards).item())
    return returns


def add_action_noise(
    env: GymEnv,
    action: torch.Tensor,
    noise_coef: float = 1.0,
) -> torch.Tensor:
    """
    adds gaussian noise to the action.
    """
    gripper_orig = action[:, -1]
    # random_upper_bound = torch.rand(size=(1,), device=action.device) * noise_coef
    noisy_action = action + torch.randn_like(action) * noise_coef  # random_upper_bound
    noisy_action = torch.clamp(
        noisy_action,
        min=torch.Tensor(env.action_space.low).to(action.device),
        max=torch.Tensor(env.action_space.high).to(action.device),
    )
    # don't want to add noise to the gripper
    env_to_check = env.envs[0].unwrapped if isinstance(env, VecEnv) else env.unwrapped
    if isinstance(env_to_check, pick_and_place.base_env.BaseEnv):
        noisy_action[:, -1] = gripper_orig
    return noisy_action


def add_observation_noise(
    env: GymEnv, observation: torch.Tensor, noise_coef: float = 1.0
) -> torch.Tensor:
    """
    adds gaussian noise to the observation.
    """
    # random_upper_bound = torch.rand(size=(1,), device=observation.device) * noise_coef
    noisy_observation = (
        observation + torch.randn_like(observation) * noise_coef
    )  # random_upper_bound
    noisy_observation = torch.clamp(
        noisy_observation,
        min=torch.Tensor(env.observation_space.low).to(observation.device),
        max=torch.Tensor(env.observation_space.high).to(observation.device),
    )
    return noisy_observation


def sr_to_reward(sr: torch.Tensor):
    """
    Convert successor representation to reward.
    :param sr: (torch.Tensor) Successor representation.
    """
    reward = torch.norm(sr, dim=1, p=2).reshape(-1, 1)
    return reward


def sr_to_value(sr: torch.Tensor):
    """
    Convert successor representation to value (think value function!).
    This interpretation is used in SRLearning.
    :param sr: (torch.Tensor) Successor representation.
    """
    return sr_to_reward(sr)


def get_rollout_subset(
    rollouts: TrajectoryDatasetSequence, fraction: float, verbose: bool = True
):
    N = int(len(rollouts) * fraction)
    assert N > 0, "Rollout Fraction too small"
    rollouts = rollouts[0:N]
    rollouts = trajectories_to_dataset(rollouts)
    rollouts = TrajectoryDatasetSequence(rollouts)
    if verbose:
        print(f"Using {len(rollouts)} rollouts.")
    return rollouts


def polyak_average(model, target_model, tau):
    """
    Polyak averaging for target networks.
    :param model: (nn.Module) Model to copy weights from.
    :param target_model: (nn.Module) Model to copy weights to.
    :param tau: (float) Polyak averaging parameter. 1.0 means hard copy.
    """
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def seed_everything(seed):
    """
    Seed everything.
    :param seed: (int) Seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def set_cuda_device(device: str):
    """
    :param device: (str) Device. e.g. "cuda:0"
    """
    torch.cuda.set_device(int(device.split(":")[-1]))


def get_wandb_name(cfg):
    """
    Get the name of the wandb run based on algorithm, environment, and seed and a 5 digit random number.
    """
    pid = os.getpid()
    algorithm = cfg.algorithm.agent._target_.split(".")[-1]
    env_name = cfg.environment.config._target_.split(".")[-1]
    wandb_name = f"{algorithm}_{env_name}_seed_{cfg.train.common.seed}_({pid})"
    return wandb_name


def combine_samples(expert_samples, rl_samples):
    if isinstance(expert_samples, ReplayBufferSamplesPlus):
        obs = torch.cat((expert_samples.observations, rl_samples.observations))
        actions = torch.cat((expert_samples.actions, rl_samples.actions))
        next_actions = torch.cat((expert_samples.next_actions, rl_samples.next_actions))
        next_obs = torch.cat(
            (expert_samples.next_observations, rl_samples.next_observations)
        )
        dones = torch.cat((expert_samples.dones, rl_samples.dones))
        rewards = torch.cat((expert_samples.rewards, rl_samples.rewards))
        return ReplayBufferSamplesPlus(
            obs, actions, next_actions, next_obs, dones, rewards
        )
    else:
        obs = torch.cat((expert_samples.observations, rl_samples.observations))
        actions = torch.cat((expert_samples.actions, rl_samples.actions))
        next_obs = torch.cat(
            (expert_samples.next_observations, rl_samples.next_observations)
        )
        dones = torch.cat((expert_samples.dones, rl_samples.dones))
        rewards = torch.cat((expert_samples.rewards, rl_samples.rewards))

        return ReplayBufferSamples(obs, actions, next_obs, dones, rewards)


def generate_offline_trajectories(
    dataset: dict,
    verbose: bool = True,
) -> Sequence[TrajectoryWithRew]:
    """Generate trajectory dictionaries D4RL offline dataset and an environment.
    Args:
    Returns:
        Sequence of trajectories
    """
    actions = dataset["actions"]
    observations = dataset["observations"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]

    trajectories = []
    trajectory_ends = np.where(np.logical_or(timeouts, terminals))[0]
    logging.info(f"Number of trajectories: {len(trajectory_ends)}")
    start_idx = 0

    for traj_end in trajectory_ends:
        # the tranjectory goes on until either the episode is done or
        # the max_time_step (which is 1000 in D4RL) is reached
        end_idx = traj_end + 1

        # NOTE: We are repeating the last observation to make the traj_obs (n+1, dim_obs)
        # where n is the number of observations in the trajectory
        # the rest (actions, rewards, ....) have (n, dim)
        # this is to make the trajectory compatible with the TrajectoryWithRew dataclass
        # which expects the observations to be (n+1, dim_obs) in order to extract the next_obs
        traj_obs = observations[start_idx:end_idx]
        traj_obs = np.concatenate([traj_obs, traj_obs[-1][None]], axis=0)
        traj_acts = actions[start_idx:end_idx]
        traj_rews = rewards[start_idx:end_idx]
        traj_terms = terminals[start_idx:end_idx]
        traj_timeouts = timeouts[start_idx:end_idx]

        # traj_infos = {k: v[start_idx:end_idx] for k, v in infos.items()}
        traj_infos = None

        assert (
            traj_terms[-1] or traj_timeouts[-1]
        ), "Trajectory should end with either a terminal state or a stimeout"

        traj = TrajectoryWithRew(
            obs=traj_obs,
            acts=traj_acts,
            rews=traj_rews,
            infos=None,
            terminal=True,  # all the trajectories are terminal
        )
        trajectories.append(traj)

        start_idx = end_idx

    if verbose:
        stats = rollout.rollout_stats(trajectories)
        logging.info(f"Rollout stats: {stats}")

    return trajectories


def get_keys(h5file):
    # from D4RL repositroy https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def download_d4rl_dataset(dataset_url):
    """
    Download the dataset given the url into demos/d4rl/datasets/
    make the directory if it doesn't exist.
    """
    # make the directory if it doesn't exist
    os.makedirs("./demos/d4rl/datasets/", exist_ok=True)
    # check if the file already exists
    filename = dataset_url.split("/")[-1]
    if os.path.exists(f"./demos/d4rl/datasets/{filename}"):
        logging.info(f"Dataset {filename} already downloaded.")
    else:
        logging.info(f"Downloading the dataset {filename}...")
        # download the dataset
        os.system(f"wget {dataset_url} -P ./demos/d4rl/datasets/")


def load_d4rl_dataset(env, env_name, dataset_type, root_path="./demos/d4rl/datasets/"):
    if env_name in ["door", "hammer", "pen", "relocate"]:
        filename = f"{env_name}-{dataset_type}.hdf5"
    elif env_name in ["ant", "halfcheetah", "hopper", "walker2d"]:
        filename = f"{env_name}_{dataset_type}.hdf5"
    h5path = os.path.join(root_path, filename)
    # from D4RL repositroy https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py
    data_dict = {}
    with h5py.File(h5path, "r") as dataset_file:

        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    # Run a few quick sanity checks
    for key in ["observations", "actions", "rewards", "terminals"]:
        assert key in data_dict, "Dataset is missing key %s" % key

    N_samples = data_dict["observations"].shape[0]
    if env.observation_space.shape is not None:
        assert (
            data_dict["observations"].shape[1:] == env.observation_space.shape
        ), "Observation shape does not match env: %s vs %s" % (
            str(data_dict["observations"].shape[1:]),
            str(env.observation_space.shape),
        )
    assert (
        data_dict["actions"].shape[1:] == env.action_space.shape
    ), "Action shape does not match env: %s vs %s" % (
        str(data_dict["actions"].shape[1:]),
        str(env.action_space.shape),
    )
    if data_dict["rewards"].shape == (N_samples, 1):
        data_dict["rewards"] = data_dict["rewards"][:, 0]

    assert data_dict["rewards"].shape == (N_samples,), "Reward has wrong shape: %s" % (
        str(data_dict["rewards"].shape)
    )
    if data_dict["terminals"].shape == (N_samples, 1):
        data_dict["terminals"] = data_dict["terminals"][:, 0]
    assert data_dict["terminals"].shape == (
        N_samples,
    ), "Terminals has wrong shape: %s" % (str(data_dict["rewards"].shape))

    return data_dict
