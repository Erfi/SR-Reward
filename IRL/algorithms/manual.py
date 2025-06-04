from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import pygame
from pygame import *


class ManualAgent(BaseAlgorithm):
    """
    Dummy agent that takes manual input from the user.
    USED ONLY FOR NavMap ENVIRONMENT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs):

        def manual_policy(*args, **kwargs):
            """
            Ask the user for an action to take using pygame keys.
            Returns: ndarray, action plus guassian noise
            """
            while True:
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LEFT]:
                            theta = -np.pi
                        elif keys[pygame.K_RIGHT]:
                            theta = 0
                        elif keys[pygame.K_UP]:
                            theta = -np.pi / 2
                        elif keys[pygame.K_DOWN]:
                            theta = np.pi / 2
                        else:
                            continue
                        action = np.array([np.cos(theta), np.sin(theta)])
                        # Make the action slightly stochastic
                        action += np.random.randn(2) * 0.1
                        # Clip the action to the range [-1, 1]
                        action = np.clip(action, -1, 1)
                        return np.expand_dims(action, axis=0), None

        return manual_policy
