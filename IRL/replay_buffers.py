from typing import Optional, List, Dict, Any, NamedTuple, Union
import numpy as np
from gymnasium import spaces
import logging
import torch

from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class SeqReplayBuffer(ReplayBuffer):
    """
    This replay buffer is used for holding sequences of observations from demonstrations.
    The transitions are added sequentially (i.e. first observation is added first).

    NOTE:
    We are changing the observation space to remove the sequence dimension
    as the observations are stored sequentially in the demonstrations.
    when we add a new transition the observation is just a single frame (not a sequence).
    and when we sample a batch, we sample a sequence of observations of whatever length we want.
    """

    def __init__(self, *args, seq_len: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.obs_shape = self.observation_space.shape[1:]
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=self.observation_space.dtype,
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=self.observation_space.dtype,
        )
        self.seq_len = seq_len
        self.is_new_traj = True
        self.starts = []
        self.ends = []
        self.global_counter = 0

        assert (self.seq_len >= 1) and isinstance(self.seq_len, int), (
            "The sequence length must be an integer greater than or equal to 1, "
            "got {} instead".format(self.seq_len)
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # if the input has sequences then we just take the last frame (the begining is history)
        if len(obs.shape) == 3:
            obs = obs[:, -1, :]
            next_obs = next_obs[:, -1, :]

        if self.full:
            logging.debug("The replay buffer is full, overriding old observations")
            # reset the position counter
            self.pos = 0
            self.full = False
        # Remove the oldest trajectory (if available and we are overriding)
        if (
            len(self.starts) > 0
            and len(self.ends) > 0
            and self.pos == self.starts[0]
            and self.global_counter > self.buffer_size - 1
        ):
            self.starts = self.starts[1:]
            self.ends = self.ends[1:]

        if self.is_new_traj:  # bookkeeping for the start and end of a trajectory
            if not done:
                self.starts.append(self.pos)
                self.is_new_traj = False
        else:
            if done:
                self.ends.append(self.pos)
                self.is_new_traj = True
                if self.ends[-1] - self.starts[-1] < self.seq_len:
                    # if the last trajectory is too short, we remove it
                    self.starts = self.starts[:-1]
                    self.ends = self.ends[:-1]

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            raise NotImplementedError(
                "Not implemented for memory efficient variant please set optimize_memory_usage=False"
            )
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        self.global_counter += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.is_new_traj = True
            # ignore the last trajectory if it is not complete
            if len(self.starts) > len(self.ends):
                self.starts = self.starts[: len(self.ends)]

    def get_obs_stats(self):
        """
        Get the mean and std of the observations.
        """
        mean = np.mean(self.observations, axis=0, keepdims=True)
        std = np.std(self.observations, axis=0, keepdims=True) + 1e-8
        return mean, std

    def normalize_obs(self, mean: np.ndarray, std: np.ndarray):
        """
        Normalize the observations.
        """
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
        include_next_actions=False,
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer with a sequence length of `self.seq_len`.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """

        if self.optimize_memory_usage:
            raise NotImplementedError(
                "Not implemented for memory efficient variant please set optimize_memory_usage=False"
            )
        if self.seq_len == 1:
            return super().sample(batch_size=batch_size, env=env)
        else:
            return self.sample_seq(
                batch_size, env=env, include_next_actions=include_next_actions
            )

    def sample_seq(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
        include_next_actions=False,
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer with a sequence length of `self.seq_len`.
        No memory efficient variant.
        """
        if self.optimize_memory_usage:
            raise NotImplementedError(
                "Not implemented for memory efficient variant please set optimize_memory_usage=False"
            )
        starts = np.array(self.starts)[: len(self.ends)]  # only completed trajectories
        ends = np.array(self.ends) - self.seq_len + 1
        if any((ends - starts) < self.seq_len):
            # This should not happen, but better safe than sorry
            short_traj_inds = np.where((ends - starts) < self.seq_len)[0]
            starts = np.delete(starts, short_traj_inds)
            ends = np.delete(ends, short_traj_inds)
            logging.warning(
                "Some trajectories are shorter than the sequence length. We won't use them."
            )
        # Sample a random sequence start
        traj_inds = np.random.randint(len(starts), size=batch_size)
        batch_inds = []
        for i in traj_inds:
            start = starts[i]  # start of the ith sampled trajectory
            end = ends[i]  # end of the ith sampled trajectory
            ind = np.random.randint(start, end + 1)  # end is inclusive (because of +1)
            batch_inds.append(ind)
        batch_inds = np.array(batch_inds)
        return self._get_seq_samples(
            batch_inds, env=env, include_next_actions=include_next_actions
        )

    def _get_seq_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
        include_next_actions=False,
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            raise NotImplementedError(
                "Not implemented for memory efficient variant please set optimize_memory_usage=False"
            )
        seq_actions = np.zeros(
            (len(batch_inds), self.seq_len, self.action_dim),
            dtype=self._maybe_cast_dtype(self.action_space.dtype),
        )
        seq_next_actions = np.zeros(
            (len(batch_inds), self.seq_len, self.action_dim),
            dtype=self._maybe_cast_dtype(self.action_space.dtype),
        )
        seq_obs = np.zeros(
            (len(batch_inds), self.seq_len, *self.obs_shape), dtype=np.float32
        )
        seq_next_obs = np.zeros(
            (len(batch_inds), self.seq_len, *self.obs_shape), dtype=np.float32
        )
        seq_dones = np.zeros((len(batch_inds), self.seq_len, 1), dtype=np.float32)
        seq_rewards = np.zeros((len(batch_inds), self.seq_len, 1), dtype=np.float32)
        for s in range(self.seq_len):
            obs = self._normalize_obs(
                self.observations[batch_inds + s, env_indices, :], env
            )
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds + s, env_indices, :], env
            )
            actions = self.actions[batch_inds + s, env_indices, :]
            rewards = self._normalize_reward(
                self.rewards[batch_inds + s, env_indices].reshape(-1, 1), env
            )
            dones = (
                self.dones[batch_inds + s, env_indices]
                * (1 - self.timeouts[batch_inds + s, env_indices])
            ).reshape(-1, 1)
            seq_obs[:, s, :] = obs
            seq_actions[:, s, :] = actions
            seq_next_obs[:, s, :] = next_obs
            seq_dones[:, s, :] = dones
            seq_rewards[:, s, :] = rewards
            if include_next_actions:
                # If we are at the end of the buffer, we use the last action
                # (it will be ignored anyway as the next observation is terminal)
                next_actions_batch_inds = np.where(
                    batch_inds + s + 1 >= len(self.actions),
                    batch_inds + s,
                    batch_inds + s + 1,
                )
                next_actions = self.actions[next_actions_batch_inds, env_indices, :]
                seq_next_actions[:, s, :] = next_actions

        if include_next_actions:
            data = (
                seq_obs,
                seq_actions,
                seq_next_actions,
                seq_next_obs,
                seq_dones,
                seq_rewards,
            )
            return ReplayBufferSamplesPlus(*tuple(map(self.to_torch, data)))
        else:
            data = (seq_obs, seq_actions, seq_next_obs, seq_dones, seq_rewards)
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class ReplayBufferSamplesPlus(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class SQILSeqReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *args,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        seq_len: int = 1,
        **kwargs,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        self.rl_buffer = SeqReplayBuffer(
            *args,
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            seq_len=seq_len,
            **kwargs,
        )
        self.expert_buffer = SeqReplayBuffer(
            *args,
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            seq_len=seq_len,
            **kwargs,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if reward == 1.0:
            self.expert_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=infos,
            )
        elif reward == 0.0:
            self.rl_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=infos,
            )
        else:
            raise ValueError("Reward must be either 0.0 or 1.0")

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
        include_next_actions=False,
    ) -> ReplayBufferSamples:
        expert_batch_size = batch_size // 2
        rl_batch_size = batch_size - expert_batch_size
        expert_samples = self.expert_buffer.sample(
            batch_size=expert_batch_size,
            env=env,
            include_next_actions=include_next_actions,
        )
        rl_samples = self.rl_buffer.sample(
            batch_size=rl_batch_size, env=env, include_next_actions=include_next_actions
        )

        samples = self.combine_samples(expert_samples, rl_samples)
        return samples

    def _get_samples(
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> ReplayBufferSamples | RolloutBufferSamples:
        pass

    def combine_samples(self, expert_samples, rl_samples):
        if isinstance(expert_samples, ReplayBufferSamplesPlus):
            obs = torch.cat((expert_samples.observations, rl_samples.observations))
            actions = torch.cat((expert_samples.actions, rl_samples.actions))
            next_actions = torch.cat(
                (expert_samples.next_actions, rl_samples.next_actions)
            )
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
