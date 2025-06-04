"""
Functions for evaluation of the models and vidualization of the results.
"""

import glob
import logging
from pathlib import Path
from datetime import datetime

import hydra
import numpy as np
from hydra import initialize, compose
from hydra.utils import get_class
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import serialize, rollout
from IRL.utils import get_rollout_subset, add_hist, sr_to_reward
import torch

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def find_all_unevaluated_models(
    model_dir: Path,
    only_completed_runs: bool = False,
):
    """
    returns a list (using glob) of paths to all the models (best_model.zip) that do not have a corresponding
    test_evaluations.npz or test_reward.npz file in the same directory.
    if only_completed_runs is True, it will only return models that have a completed training log.
    """
    all_zip_files = glob.glob(str(model_dir / f"**/best_model.zip"), recursive=True)
    zip_files = []
    for zip_file in all_zip_files:
        if not Path(zip_file).with_name("test_evaluations.npz").exists():
            if only_completed_runs:
                data = np.load(Path(zip_file).with_name("evaluations.npz"))
                if (
                    data["timesteps"][-1] >= 1e6
                ):  # 1e6 is the number of timesteps in the training
                    zip_files.append(Path(zip_file))
                else:
                    logging.info(
                        f"Skipping {zip_file} as it is not a completed run (timesteps: {data['timesteps'][-1]})"
                    )
            else:
                zip_files.append(Path(zip_file))
        else:
            logging.info(
                f"Skipping {zip_file} as it already has a test_evaluations.npz file"
            )
    return zip_files


def remove_incomplete_runs(model_dir: Path):
    """
    Removes all the models (and their corresponding folder) if there are less than 1e6 timesteps in the training.
    """
    all_evaluations = glob.glob(str(model_dir / f"**/evaluations.npz"), recursive=True)
    for evaluation in all_evaluations:
        data = np.load(evaluation)
        if data["timesteps"][-1] < 1e6:
            logging.info(
                f"Removing the folder containing {evaluation} as it is an incomplete run"
            )
            folder = Path(evaluation).parent
            for file in folder.glob("*"):
                file.unlink()
            folder.rmdir()
    return


def find_all_evaluated_models(model_dir: Path):
    """
    returns a list (using glob) of paths to all the models (best_model.zip) that have a corresponding
    test_evaluations.npz file in the same directory.
    """
    all_zip_files = glob.glob(str(model_dir / f"**/best_model.zip"), recursive=True)
    zip_files = []
    for zip_file in all_zip_files:
        if Path(zip_file).with_name("test_evaluations.npz").exists():
            zip_files.append(Path(zip_file))
        else:
            logging.info(
                f"Skipping {zip_file} as it does not have a test_evaluations.npz"
            )
    return zip_files


def get_all_evaluations(model_paths: list[Path], save_path: Path = None):
    """
    Reads the test_evaluations.npz files from the model directories and prints the evaluation results.
    """
    if save_path is not None:
        f = open(save_path, "w")

    for model_path in model_paths:
        info = get_info_from_path(model_path)
        # recompose the config for this env and this algorithm
        config = recompose_config(
            [
                f"algorithm={info['algorithm']}",
                f"environment={info['env_name'].lower()}",
            ]
        )
        random_score = config.environment.d4rl.min_score
        expert_score = config.environment.d4rl.max_score
        expert_normalized = normalize_reward(expert_score, expert_score, random_score)
        data = np.load(Path(model_path).parent / "test_evaluations.npz")
        mean_reward = data["mean_reward"]
        std_reward = data["std_reward"]
        normalized_reward = normalize_reward(mean_reward, expert_score, random_score)
        normalized_std = normalize_reward(std_reward, expert_score, random_score)
        n_eval_episodes = data["n_eval_episodes"]
        env = info["env_name"]
        algo = info["algorithm"]
        seed = info["seed"]
        exp = info["experiment_name"]
        logging.info("--------------------")
        logging.info(
            f"Env: {env} | Algo: {algo} | Exp: {exp} | Seed: {seed} | Mean Reward: {normalized_reward} | Std Reward: {normalized_std} | n_eval_episodes: {n_eval_episodes} | Expert Score: {expert_normalized}"
        )
        if save_path is not None:
            f.write(
                f"Env: {env} | Algo: {algo} | Exp: {exp} | Seed: {seed} | Mean Reward: {normalized_reward} | Std Reward: {normalized_std} | n_eval_episodes: {n_eval_episodes}| Expert Score: {expert_normalized} \n"
            )
    if save_path is not None:
        f.close()


def normalize_reward(rewards, expert_score, random_score):
    return 100 * (rewards - random_score) / (expert_score - random_score)


def recompose_config(overrides: list[str]):
    """
    recompose the config with the overrides
    we can use this to get the config for the model evaluation
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="config", job_name="model_evaluation")
    config = compose(config_name="config", overrides=overrides)
    return config


def evaluate_model(model_path: Path, n_eval_episodes: int = 10):
    logging.info(f"Evaluating model: {model_path}")
    info = get_info_from_path(model_path)
    # recompose the config for this env and this algorithm
    config = recompose_config(
        [
            f"algorithm={info['algorithm']}",
            f"environment={info['env_name'].lower()}",
        ]
    )
    try:
        # load the environment
        env = hydra.utils.instantiate(config.environment.config)
        # Load the agent
        agent = get_class(config.algorithm.agent._target_).load(
            str(model_path),
            print_system_info=False,
        )
        # Evaluate the agent
        rewards, eps_len = evaluate_policy(
            model=agent,
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            return_episode_rewards=True,
        )
        # save the evaluation results in the same directory as the model
        np.savez(
            Path(model_path).parent / "test_evaluations.npz",
            mean_reward=np.mean(rewards),
            std_reward=np.std(rewards),
            rewards=np.array(rewards),
            eps_len=np.array(eps_len),
            n_eval_episodes=n_eval_episodes,
        )
    except (RuntimeError, TypeError, ValueError) as e:
        logging.error(f"Error evaluating model: {model_path}")
        logging.error(e)


def get_info_from_path(model_path: Path):
    """
    Extracts the algorithm name, environment name, experiment name, seed, run time and data from the model path.
    returns a dictionary of information
    """
    parts = model_path.parts
    algorithm = parts[1]
    env = parts[2]
    experiment_name = parts[3]
    seed = int(parts[4].split("_")[-1])
    date_time = parts[5].split("__")[0]
    run_datetime = datetime.strptime(date_time, "%y%m%d_%H%M%S")
    data = {
        "algorithm": algorithm,
        "env_name": env,
        "experiment_name": experiment_name,
        "seed": seed,
        "run_datetime": run_datetime,
    }
    return data


def evaluate_reward(model_path: Path, n_noise_levels: int = 11):
    logging.info(f"Evaluating reward function for trained model: {model_path}")
    info = get_info_from_path(model_path)
    # recompose the config for this env and this algorithm
    config = recompose_config(
        [
            f"algorithm={info['algorithm']}",
            f"environment={info['env_name'].lower()}",
        ]
    )
    rollouts = serialize.load(config.memory.saved_memory_filename)
    rollouts = get_rollout_subset(rollouts, config.train.common.data_fraction)

    noises = np.linspace(0, 1, n_noise_levels)
    # Load the agent from model path
    agent = get_class(config.algorithm.agent._target_).load(
        str(model_path),
        print_system_info=False,
    )

    visitnet = agent.visitnet
    visitnet.eval()
    visitnet.to(config.train.common.device)
    visitnet.requires_grad_(False)

    h = config.environment.config.history_len
    mean_returns = {}

    for noise in noises:
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
        mean_returns.update({noise: np.mean(returns)})

    # print pretty upto 5 decimal places
    logging.info(f"Evaluated reward using {len(rollouts)} rollouts")
    for noise, mean_return in mean_returns.items():
        logging.info(f"Noise: {noise:.5f}, Mean Return: {mean_return:.5f}")

    # save the evaluation results in the same directory as the model
    np.savez(
        Path(model_path).parent / "test_reward.npz",
        rewards=list(mean_returns.values()),
        noises=list(mean_returns.keys()),
        # n_noise_levels=n_noise_levels,
    )


def evaluate_trajectory_reward(model_path: Path):
    logging.info(
        f"Evaluating reward function for trained model: {model_path} on expert trajectories"
    )
    info = get_info_from_path(model_path)

    # Recompose the config for this environment and algorithm
    config = recompose_config(
        [
            f"algorithm={info['algorithm']}",
            f"environment={info['env_name'].lower()}",
        ]
    )

    # Load rollouts
    rollouts = serialize.load(config.memory.saved_memory_filename)
    rollouts = get_rollout_subset(rollouts, config.train.common.data_fraction)

    # Load the agent from the model path
    agent = get_class(config.algorithm.agent._target_).load(
        str(model_path),
        print_system_info=False,
    )

    visitnet = agent.srnet
    visitnet.eval()
    visitnet.to(config.train.common.device)
    visitnet.requires_grad_(False)

    h = config.environment.config.history_len
    instant_rewards = []

    for rollout in rollouts:
        obs = add_hist(rollout.obs[:-1], h=h)
        acts = rollout.acts[h - 1 :]
        obs = torch.from_numpy(obs).float().to(visitnet.device)
        acts = torch.from_numpy(acts).float().to(visitnet.device)

        with torch.no_grad():
            sr = visitnet(obs, acts)

        rewards = sr_to_reward(sr)
        instant_rewards.append(rewards.cpu().numpy().reshape(-1).tolist())

    # Save the instant rewards
    np.savez(
        model_path.parent / "instant_rewards_on_expert_traj.npz",
        instant_rewards=instant_rewards,
    )

    logging.info(
        f"Evaluated instant rewards on expert trajectories using {len(rollouts)} rollouts"
    )
    # return instant_rewards
