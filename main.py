import os
import logging
from datetime import datetime

import cv2
import numpy as np
import wandb
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import serialize

from IRL.utils import (
    seed_everything,
    set_cuda_device,
    get_wandb_name,
    get_rollout_subset,
    download_d4rl_dataset,
    load_d4rl_dataset,
    generate_offline_trajectories,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode.startswith("train"):
        PID = os.getpid()
        logging.info(f"Process ID: {PID}")

        cfg.algorithm.saved_model_timestamp = (
            f"/{datetime.now().strftime('%y%m%d_%H%M%S')}"
        )

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb_name = get_wandb_name(cfg)
        wandb.init(name=wandb_name, **cfg.wb, config=cfg_dict)

    rng = np.random.default_rng(cfg.environment.config.seed)
    seed_everything(cfg.environment.config.seed)
    set_cuda_device(cfg.train.common.device)

    if cfg.mode == "train_rl":
        logging.info("Training the RL agent...")
        cfg.algorithm.saved_model_timestamp = (
            f"{cfg.algorithm.saved_model_timestamp}__{PID}"
        )
        env = hydra.utils.instantiate(cfg.environment.config)
        eval_env = hydra.utils.instantiate(cfg.environment.config)
        agent = hydra.utils.instantiate(
            config=cfg.algorithm.agent,
            env=env,
            _convert_="partial",
        )
        eval_callback = hydra.utils.instantiate(
            config=cfg.train.eval_callback, eval_env=eval_env
        )
        agent.learn(
            total_timesteps=cfg.train.common.train_steps,
            log_interval=cfg.train.common.train_log_interval,
            callback=eval_callback,
        )
        logging.info("RL agent training complete.")

    elif cfg.mode == "train_irl":
        logging.info("Training IRL Agent...")
        cfg.algorithm.saved_model_timestamp = (
            f"{cfg.algorithm.saved_model_timestamp}__{PID}"
        )
        env = hydra.utils.instantiate(cfg.environment.config)
        eval_env = hydra.utils.instantiate(cfg.environment.config)
        rollouts = serialize.load(cfg.memory.saved_memory_filename)
        rollouts = get_rollout_subset(rollouts, cfg.train.common.data_fraction)

        if cfg.memory.extra_memory_filename:
            extra_rollouts = serialize.load(cfg.memory.extra_memory_filename)
            extra_rollouts = get_rollout_subset(
                extra_rollouts, cfg.train.common.data_fraction
            )
            rollouts = {"expert_rollouts": rollouts, "extra_rollouts": extra_rollouts}
        agent = hydra.utils.instantiate(
            cfg.algorithm.agent,
            env=env,
            demonstrations=rollouts,
            _convert_="partial",
        )
        eval_callback = hydra.utils.instantiate(
            cfg.train.eval_callback, eval_env=eval_env
        )
        agent.learn(
            total_timesteps=cfg.train.common.train_steps,
            log_interval=cfg.train.common.train_log_interval,
            callback=eval_callback,
        )
        logging.info("IRL agent training complete.")

    elif cfg.mode == "evaluate":
        logging.info("Evaluating the agent...")
        eval_env = hydra.utils.instantiate(cfg.environment.config)
        agent = get_class(cfg.algorithm.agent._target_).load(
            cfg.algorithm.saved_model_path
        )
        reward_mean, reward_std = evaluate_policy(
            model=agent,
            env=eval_env,
            n_eval_episodes=cfg.train.common.eval_episodes,
            deterministic=True,
            render=True,
            return_episode_rewards=False,
        )
        logging.info(f"Reward: {reward_mean:.2f}+/-{reward_std:.2f}.")
        logging.info("RL agent evaluation complete.")

    elif cfg.mode == "make_video":
        logging.info("Making a video...")
        os.makedirs("videos", exist_ok=True)
        env = hydra.utils.instantiate(cfg.environment.config)
        agent = get_class(cfg.algorithm.agent._target_).load(
            cfg.algorithm.saved_model_path
        )
        frames = []
        n_episodes = 5
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                frame = env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminal, terminated, _ = env.step(action)
                done = terminal or terminated

        height, width, layers = frames[0].shape
        video_name = f"./videos/{cfg.environment.meta.name}_video.mp4"
        video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
        )

        for frame in frames:
            video.write(frame)

        video.release()
        logging.info(f"Video saved as {video_name}")

    elif cfg.mode == "d4rl_data":
        logging.info("loading d4rl data...")
        assert (
            cfg.environment.config.history_len == 1
        ), "For D4RL data conversion set history_len to 1"
        env = hydra.utils.instantiate(cfg.environment.config)
        env_name = cfg.environment.d4rl.env_name
        dataset_type = cfg.environment.d4rl.dataset_type
        dataset_url = cfg.environment.d4rl[f"dataset_{dataset_type}_url"]
        download_d4rl_dataset(dataset_url=dataset_url)
        dataset = load_d4rl_dataset(
            env=env, env_name=env_name, dataset_type=dataset_type
        )
        rollouts = generate_offline_trajectories(
            dataset,
            verbose=True,
        )
        name = f"{env_name}_{dataset_type}"
        serialize.save(f"./demos/{name}_demo.pkl", rollouts)
        logging.info("Offline D4RL demos saved")


if __name__ == "__main__":
    main()
