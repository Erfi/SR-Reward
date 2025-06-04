"""
Base class for all IRL algorithms.
"""

import logging
from typing import Any, Optional, Tuple, Union, Dict, Type, TypeVar, List
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from imitation.algorithms.base import AnyTransitions
from imitation.data import types as demo_types
from imitation.data import rollout
from imitation.util import util

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

SelfBaseIRLAgent = TypeVar("SelfBaseIRLAgent", bound="BaseIRLAgent")


class BaseIRLAgent(BaseAlgorithm):
    """
    The base class for all IRL algorithms.
    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from
        demonstrations: The demonstrations for the agent
        learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        policy_kwargs: Additional arguments to be passed to the policy on creation
        stats_window_size:  Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: The log location for tensorboard (if None, no logging)
        verbose: The verbosity level: 0 none, 1 training information, 2 debug
        device: Device on which the code should run.
            By default, it will try to use the GPU if it is available.
            Can be "cpu", "cuda", "auto" or a torch.device object.
        support_multi_env: Whether the algorithm supports training
            with multiple environments simultaneously.
        monitor_wrapper: Whether to wrap the environment in a Monitor wrapper.
        seed: Seed for the pseudo random generators (python, numpy, pytorch).
        use_sde: Whether to use generalized State Dependent Exploration
            (gSDE) instead of action noise exploration (default: False).
            This can improve performance for some tasks.
        sde_sample_freq: Sample a new noise matrix every n steps when using gSDE.
            This is irrelevant when not using gSDE (default: -1).
        supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: GymEnv,
        demonstrations: AnyTransitions | None,
        learning_rate: Union[float, Schedule],
        batch_size: int = 256,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        offline_replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        offline_replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        online_replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        online_replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: str = None,
        verbose: int = 0,
        device: str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: int = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.offline_replay_buffer_class = offline_replay_buffer_class
        self.offline_replay_buffer_kwargs = offline_replay_buffer_kwargs
        self.online_replay_buffer_class = online_replay_buffer_class
        self.online_replay_buffer_kwargs = online_replay_buffer_kwargs
        self.batch_size = batch_size
        self.demonstrations = demonstrations

    def _setup_model(self) -> None:
        """
        Set up replay buffer and policy for IRL algorithms.
        """
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        if self.online_replay_buffer_class is not None:
            self.setup_interaction()

        # convert demonstrations to replay buffer
        if self.demonstrations is not None:
            self.set_demonstrations(self.demonstrations)
        else:
            logging.warning(
                "No demonstrations provided. This is only okay for evaluation."
            )

    def setup_interaction(self) -> None:
        self.online_replay_buffer = self.online_replay_buffer_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            **self.online_replay_buffer_kwargs,
        )
        self.action_scale = (self.action_space.high - self.action_space.low) / 2.0

    def set_demonstrations(self, demonstrations: AnyTransitions) -> None:
        """
        Set the demonstrations for the agent.
        These can be rollouts or transitions or be made into a replay buffer.
        modifed from: https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/sqil.py
        """
        if not isinstance(demonstrations, demo_types.Transitions):
            (
                item,
                demonstrations,
            ) = util.get_first_iter_element(  # type: ignore[assignment]
                demonstrations,  # type: ignore[assignment]
            )
            if isinstance(item, demo_types.TrajectoryWithRew):
                demonstrations = rollout.flatten_trajectories_with_rew(demonstrations)
            elif isinstance(item, demo_types.Trajectory):
                demonstrations = rollout.flatten_trajectories(demonstrations)

        if not isinstance(demonstrations, demo_types.Transitions):
            raise NotImplementedError(
                f"Unsupported demonstrations type: {demonstrations}",
            )

        n_samples = len(demonstrations)
        self.offline_replay_buffer = self.offline_replay_buffer_class(
            buffer_size=n_samples,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            **self.offline_replay_buffer_kwargs,
        )
        for transition in demonstrations:
            self.offline_replay_buffer.add(
                obs=transition["obs"],
                next_obs=transition["next_obs"],
                action=transition["acts"],
                done=transition["dones"],
                reward=transition.get("rews", np.inf),
                infos=[transition.get("infos", {})],
            )

    def learn(
        self: SelfBaseIRLAgent,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBaseIRLAgent:
        """
        The main training loop for IRL algorithms.
        """
        self.log_interval = log_interval
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())
        if self.online_replay_buffer_class is not None:
            assert (
                self.env is not None
            ), "You must set the environment before calling learn() (required for online replay buffer)"

        while self.num_timesteps < total_timesteps:
            self.interact()
            self.train(batch_size=self.batch_size)
            self.num_timesteps += 1
            callback.update_locals(locals())
            callback.on_step()
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
        callback.on_training_end()

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the parameters that should not be saved.
        We don't want to same the demonstrations they take up too much space.
        """
        return super()._excluded_save_params() + ["demonstrations"]

    def train(self, batch_size: int) -> None:
        """
        The training loop for IRL algorithms.
        Sample the replay buffer and update the networks.
        Also update the target networks if applicable.
        """
        raise NotImplementedError()

    def interact(self) -> None:
        """
        The interaction loop for IRL algorithms.
        Sample the environment and add the samples to the replay buffer.
        """
        if self.online_replay_buffer_class is not None:
            raise NotImplementedError(
                "Interaction loop not implemented for online replay buffer.",
            )
        else:
            pass
