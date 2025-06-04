"""
Implementation of Behavioral Cloning (BC) algorithm.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from imitation.algorithms.base import AnyTransitions

from IRL.algorithms.base import BaseIRLAgent
from IRL.networks import IRLActor, BaseSeqFeaturesExtractor


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
SelfBCAgent = TypeVar("SelfBCAgent", bound="BCAgent")


class BCPolicy(BasePolicy):
    """
    NOTE: Usially the same optimizer is used for all the parameters of the policy,
    but here we are using different optimizers (perhaps with different LR,..) for different networks.
    Hence the self.optimizer_kwargs attribute are not used.
    self.optimizer_class is used as they are all the same class (Adam).
    """

    actor: IRLActor

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        actor_kwargs: Optional[Dict[str, Any]] = None,
        actor_optim_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        squash_output: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )
        self.actor_kwargs = actor_kwargs or {}
        self.actor_optim_kwargs = actor_optim_kwargs or {}
        self._build()

    def _build(self) -> None:
        """
        Build all the networks and optimizers
        """
        # not sharing the features extractors
        actor_features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )

        # Actor Network
        self.actor = IRLActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=actor_features_extractor,
            features_dim=actor_features_extractor.features_dim,
            **self.actor_kwargs,
        ).to(self.device)

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            **self.actor_optim_kwargs,
        )
        # Set networks to training mode (except for target networks)
        self.actor.set_training_mode(True)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.training = mode


class BCAgent(BaseIRLAgent):
    policy: BCPolicy
    actor: IRLActor

    def __init__(
        self,
        policy: Type[BCPolicy],
        env: GymEnv,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = ReplayBuffer,
        demonstrations: AnyTransitions = None,
        batch_size: int = 256,
        learning_rate: Union[float, Schedule] = 3e-4,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            demonstrations=demonstrations,
            learning_rate=learning_rate,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            offline_replay_buffer_class=replay_buffer_class,
            offline_replay_buffer_kwargs=replay_buffer_kwargs,
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=False,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            supported_action_spaces=(gym.spaces.Box,),
        )
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor

    def learn(
        self: SelfBCAgent,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBCAgent:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, batch_size: int) -> None:
        sample = self.offline_replay_buffer.sample(
            batch_size, env=self._vec_normalize_env
        )
        state = sample.observations
        if issubclass(self.policy.features_extractor_class, BaseSeqFeaturesExtractor):
            action = sample.actions[:, -1, :]  # use the last action
        else:
            action = sample.actions
        actornet_loss = self._update_actor_network(
            state,
            action,
        )

        # Training Log
        if self.num_timesteps % self.log_interval == 0:
            self.logger.record("train/actornet_loss", actornet_loss)
            self.logger.dump(step=self.num_timesteps)

    def _update_actor_network(self, state, action):
        self.actor.set_training_mode(True)
        logprob = self.actor.action_log_prob(obs=state, action=action)
        logprob = logprob.reshape(-1, 1)  # reshape to (batch_size, 1)

        loss = -logprob.mean()
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        self.actor.set_training_mode(False)
        return loss.item()
