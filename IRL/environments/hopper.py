import gymnasium as gym

from IRL.environments.wrappers import GymHistoryWrapper


class Hopper(gym.Wrapper):
    def __init__(
        self,
        render_mode=None,
        max_episode_steps=1000,
        history_len=1,
        seed=0,
        **kwargs,
    ):
        env = gym.make(
            "Hopper-v2",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        env = GymHistoryWrapper(env, history_len=history_len, seed=seed)
        super().__init__(env)
