import gymnasium as gym
import mani_skill2.envs  # otherwise can't find PickCube-v0

from IRL.environments.wrappers import ManiSkillWrapper


class PickCube(gym.Wrapper):
    def __init__(
        self,
        obs_mode,
        control_mode,
        render_mode=None,
        max_episode_steps=200,
        history_len=1,
        seed=0,
        **kwargs,
    ):
        env = gym.make(
            "PickCube-v0",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            obs_mode=obs_mode,
            control_mode=control_mode,
            **kwargs,
        )
        env = ManiSkillWrapper(env, history_len=history_len, seed=seed)
        super().__init__(env)
