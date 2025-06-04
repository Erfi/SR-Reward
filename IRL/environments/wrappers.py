from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym


class ManiSkillWrapper(gym.Wrapper):
    """
    Wrapper for ManiSkill environments to make them include history of observations.
    In addition, it deals with image observations by converting them to RGBD.
    """

    def __init__(self, env, history_len=1, seed=0):
        self.obs_mode = env.unwrapped.obs_mode
        if self.obs_mode == "rgbd":
            width = env.unwrapped._camera_cfgs["base_camera"].width
            height = env.unwrapped._camera_cfgs["base_camera"].height
            env.observation_space = gym.spaces.Box(
                low=0, high=1.0, shape=(height, width, 8), dtype=np.float32
            )
            self.base_near = env.unwrapped._camera_cfgs["base_camera"].near
            self.base_far = env.unwrapped._camera_cfgs["base_camera"].far
            self.hand_near = env.unwrapped._camera_cfgs["hand_camera"].near
            self.hand_far = env.unwrapped._camera_cfgs["hand_camera"].far
        self.history_len = history_len
        self.observation_history = deque(maxlen=history_len)
        if self.history_len > 1:
            env.observation_space = self._expand_observation_space(
                env.observation_space
            )

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        super().__init__(env)

    def _expand_observation_space(self, observation_space):
        if self.history_len > 1:
            return gym.spaces.Box(
                low=np.repeat(
                    np.expand_dims(observation_space.low, axis=0),
                    self.history_len,
                    axis=0,
                ),
                high=np.repeat(
                    np.expand_dims(observation_space.high, axis=0),
                    self.history_len,
                    axis=0,
                ),
                shape=(self.history_len, *observation_space.shape),
                dtype=np.float32,
            )
        else:
            return self.observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.observation_history.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.observation(obs)

        if self.history_len > 1:
            for _ in range(self.history_len):
                self.observation_history.append(obs)
            obs = np.array(self.observation_history)
        return obs, info

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        reward = self.reward(reward)
        obs = self.observation(obs)
        if self.history_len > 1:
            self.observation_history.append(obs)
            obs = np.array(self.observation_history)
        return obs, reward, terminal, truncated, info

    def reward(self, reward):
        if self.unwrapped.reward_mode == "sparse" and reward == 1:
            reward += 1.0 - (self.unwrapped.elapsed_steps / self.spec.max_episode_steps)
        return reward

    def observation(self, obs):
        if self.obs_mode == "state":
            return obs
        elif self.obs_mode == "rgbd":
            obs_base_rgb = np.array(obs["image"]["base_camera"]["rgb"]) / 255.0
            obs_hand_rgb = np.array(obs["image"]["hand_camera"]["rgb"]) / 255.0
            obs_base_depth = np.array(obs["image"]["base_camera"]["depth"])
            obs_hand_depth = np.array(obs["image"]["hand_camera"]["depth"])
            obs_base_depth = self._normalize_depth(
                obs_base_depth, near=self.base_near, far=self.base_far
            )
            obs_hand_depth = self._normalize_depth(
                obs_hand_depth, near=self.hand_near, far=self.hand_far
            )
            observation = np.concatenate(
                [obs_base_rgb, obs_base_depth, obs_hand_rgb, obs_hand_depth],
                axis=-1,
                dtype=np.float32,
            )
            return observation

    def _normalize_depth(self, depth, near, far):
        depth = (depth - near) / (far - near)
        depth = np.clip(depth, 0, 1)
        return depth


class GymHistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_len=1, seed=0):
        self.history_len = history_len
        self.observation_history = deque(maxlen=history_len)
        if self.history_len > 1:
            env.observation_space = self._expand_observation_space(
                env.observation_space
            )

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        super().__init__(env)

    def _expand_observation_space(self, observation_space):
        if self.history_len > 1:
            return gym.spaces.Box(
                low=np.repeat(
                    np.expand_dims(observation_space.low, axis=0),
                    self.history_len,
                    axis=0,
                ),
                high=np.repeat(
                    np.expand_dims(observation_space.high, axis=0),
                    self.history_len,
                    axis=0,
                ),
                shape=(self.history_len, *observation_space.shape),
                dtype=np.float32,
            )
        else:
            return self.observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.observation_history.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.observation(obs)

        if self.history_len > 1:
            for _ in range(self.history_len):
                self.observation_history.append(obs)
            obs = np.array(self.observation_history)
        return obs, info

    def observation(self, obs):
        return obs

    def reward(self, reward):
        return reward

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        reward = self.reward(reward)
        obs = self.observation(obs)
        if self.history_len > 1:
            self.observation_history.append(obs)
            obs = np.array(self.observation_history)
        return obs, reward, terminal, truncated, info
