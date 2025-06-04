"""
This is the implementation of Dual V-Learning (DVL) algorithm based on
[1] Dual RL: Unification and New Methods for Reinforcement and Imitation Learning Harshit Sikchi, Qinqing Zheng, Amy Zhang, Scott Niekum;
It provides alternative choices to the Gumbel regression loss that fixes the known training instability issue of XQL[2].
"""

import logging
from typing import Any, Dict, Optional, Union, Type, TypeVar

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from imitation.algorithms.base import AnyTransitions

from IRL.utils import sr_to_reward, add_action_noise, add_observation_noise
from IRL.algorithms.base import BaseIRLAgent
from IRL.networks import (
    SRNet,
    ValueNet,
    ContinuousCritic,
    IRLActor,
    BaseSeqFeaturesExtractor,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

SelffDVLAgent = TypeVar("SelffDVLAgent", bound="fDVLAgent")


class fDVLPolicy(BasePolicy):
    """
    NOTE: Usially the same optimizer is used for all the parameters of the policy,
    but here we are using different optimizers (perhaps with different LR,..) for different networks.
    Hence the self.optimizer_kwargs attribute are not used.
    self.optimizer_class is used as they are all the same class (Adam).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        visitnet_features_extractor_class: Type[
            BaseFeaturesExtractor
        ] = FlattenExtractor,
        visitnet_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        visitnet_class: Type[BaseModel] = SRNet,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        critic_kwargs: Optional[Dict[str, Any]] = None,
        actor_kwargs: Optional[Dict[str, Any]] = None,
        valuenet_kwargs: Optional[Dict[str, Any]] = None,
        visitnet_kwargs: Optional[Dict[str, Any]] = None,
        critic_optim_kwargs: Optional[Dict[str, Any]] = None,
        actor_optim_kwargs: Optional[Dict[str, Any]] = None,
        valuenet_optim_kwargs: Optional[Dict[str, Any]] = None,
        visitnet_optim_kwargs: Optional[Dict[str, Any]] = None,
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
        self.critic_kwargs = critic_kwargs or {}
        self.actor_kwargs = actor_kwargs or {}
        self.valuenet_kwargs = valuenet_kwargs or {}
        self.visitnet_kwargs = visitnet_kwargs or {}
        self.critic_optim_kwargs = critic_optim_kwargs or {}
        self.actor_optim_kwargs = actor_optim_kwargs or {}
        self.valuenet_optim_kwargs = valuenet_optim_kwargs or {}
        self.visitnet_optim_kwargs = visitnet_optim_kwargs or {}
        # visitnet has it's own class and features extractor class
        self.visitnet_class = visitnet_class
        self.visitnet_features_extractor_class = visitnet_features_extractor_class
        self.visitnet_features_extractor_kwargs = (
            visitnet_features_extractor_kwargs or {}
        )
        self._build()

    def _build(self) -> None:
        """
        Build all the networks and optimizers
        """
        # not sharing the features extractors
        actor_features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        critic_features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        critic_target_features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        valuenet_features_extractor = self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        visitnet_features_extractor = self.visitnet_features_extractor_class(
            self.observation_space, **self.visitnet_features_extractor_kwargs
        )

        # Critic Network
        self.critic = ContinuousCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=critic_features_extractor,
            features_dim=critic_features_extractor.features_dim,
            **self.critic_kwargs,
        ).to(self.device)
        # Critic Target Network
        self.critic_target = ContinuousCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=critic_target_features_extractor,
            features_dim=critic_target_features_extractor.features_dim,
            **self.critic_kwargs,
        ).to(self.device)
        # Actor Network
        self.actor = IRLActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=actor_features_extractor,
            features_dim=actor_features_extractor.features_dim,
            **self.actor_kwargs,
        ).to(self.device)
        # Value Network
        self.valuenet = ValueNet(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=valuenet_features_extractor,
            features_dim=valuenet_features_extractor.features_dim,
            **self.valuenet_kwargs,
        ).to(self.device)
        # VisitNet Network
        self.visitnet = self.visitnet_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=visitnet_features_extractor,
            features_dim=visitnet_features_extractor.features_dim,
            **self.visitnet_kwargs,
        ).to(self.device)

        # Optimizers
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            **self.critic_optim_kwargs,
        )
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            **self.actor_optim_kwargs,
        )
        self.valuenet.optimizer = self.optimizer_class(
            self.valuenet.parameters(),
            **self.valuenet_optim_kwargs,
        )
        self.visitnet.optimizer = self.optimizer_class(
            self.visitnet.parameters(),
            **self.visitnet_optim_kwargs,
        )

        # Set parameters of target networks to be the same as the online networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Set networks to trianing mode
        self.critic.set_training_mode(True)
        self.actor.set_training_mode(True)
        self.valuenet.set_training_mode(True)
        self.visitnet.set_training_mode(True)
        # set target networks to eval mode
        self.critic_target.set_training_mode(False)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.train(mode)
        self.valuenet.train(mode)
        self.visitnet.train(mode)
        self.training = mode


class fDVLAgent(BaseIRLAgent):
    def __init__(
        self,
        policy: Type[fDVLPolicy],
        env: GymEnv,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = ReplayBuffer,
        demonstrations: AnyTransitions = None,
        batch_size: int = 256,
        learning_rate: Union[float, Schedule] = 3e-4,
        norm_loss_coef: float = 0.0,
        use_ground_truth_reward: bool = False,
        reconstruction_loss_coef: float = 0.0,
        neg_sampling_loss_coef: float = 0.0,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ns_noise_frac: float = 0.0,
        ns_noise_penalty_std: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,
        lambda_: float = 0.5,
        pretrain_visitnet_steps: int = 1000,
        value_loss_fn: str = "chi-square",
        max_clip: float = 20.0,
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
            supported_action_spaces=(spaces.Box,),
        )
        self.use_ground_truth_reward = use_ground_truth_reward
        self.ns_noise_frac = ns_noise_frac
        self.ns_noise_penalty_std = ns_noise_penalty_std
        self.norm_loss_coef = norm_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.neg_sampling_loss_coef = neg_sampling_loss_coef
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lambda_ = lambda_
        self.pretrain_visitnet_steps = pretrain_visitnet_steps
        self.value_loss_fn = value_loss_fn
        self.max_clip = max_clip
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.valuenet = self.policy.valuenet
        self.visitnet = self.policy.visitnet

    def learn(
        self: SelffDVLAgent,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "fDVL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelffDVLAgent:
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
            batch_size, env=self._vec_normalize_env, include_next_actions=True
        )
        observations = sample.observations
        next_observations = sample.next_observations
        if issubclass(self.policy.features_extractor_class, BaseSeqFeaturesExtractor):
            actions = sample.actions[:, -1, :]  # use the last action
            next_actions = sample.next_actions[:, -1, :]  # use the last next action
            dones = sample.dones[:, -1, :]  # use the last done
            rewards = sample.rewards[:, -1, :]  # use the last reward
        else:
            actions = sample.actions
            next_actions = sample.next_actions
            dones = sample.dones
            rewards = sample.rewards

        perturbed_observations, perturbed_actions = None, None
        if self.use_ground_truth_reward:
            visitnet_loss = {
                "loss": 0.0,
                "bellman_loss": 0.0,
                "reconstruction_loss": 0.0,
                "norm_loss": 0.0,
                "ns_loss": 0.0,
                "exp_decay_weight": 0.0,
            }
        else:
            if self.ns_noise_frac > 0.0:
                perturbed_observations = add_observation_noise(
                    self.env, observations.clone(), noise_coef=self.ns_noise_frac
                )
                perturbed_actions = add_action_noise(
                    self.env, actions.clone(), noise_coef=self.ns_noise_frac
                )
            visitnet_loss = self._update_visitnet(
                observations,
                actions,
                dones,
                next_observations,
                next_actions,
                perturbed_observations,
                perturbed_actions,
            )
        if self.num_timesteps > self.pretrain_visitnet_steps:
            if perturbed_observations is not None and perturbed_actions is not None:
                assert (
                    self.use_ground_truth_reward == False
                ), "cannot use ground truth reward with negative sampling"
                observations = torch.cat([observations, perturbed_observations], dim=0)
                actions = torch.cat([actions, perturbed_actions], dim=0)
                next_observations = torch.cat(
                    [next_observations, next_observations], dim=0
                )
                dones = torch.cat([dones, dones], dim=0)
            criticnet_loss = self._update_critic_network(
                observations,
                next_observations,
                actions,
                dones,
                reward=rewards if self.use_ground_truth_reward else None,
            )
            valuenet_loss = self._update_value_network(
                observations,
                actions,
            )
            actornet_loss = self._update_actor_network(
                observations,
                actions,
            )
        else:
            criticnet_loss = {
                "loss": 0.0,
                "mean_reward": 0.0,
                "ns_loss": 0.0,
                "exp_decay_weight": 0.0,
            }
            actornet_loss = {"loss": 0.0}
            valuenet_loss = {"loss": 0.0, "q_minus_v_mean": 0.0, "ns_loss": 0.0}

        # Training Log
        if self.num_timesteps % self.log_interval == 0:
            self.logger.record("train/visitnet_loss", visitnet_loss["loss"])
            self.logger.record(
                "train/visitnet_bellman_loss", visitnet_loss["bellman_loss"]
            )
            self.logger.record(
                "train/visitnet_reconstruction_loss",
                visitnet_loss["reconstruction_loss"],
            )
            self.logger.record("train/visitnet_norm_loss", visitnet_loss["norm_loss"])
            self.logger.record("train/visitnet_ns_loss", visitnet_loss["ns_loss"])
            self.logger.record(
                "train/visitnet_exp_decay_weight", visitnet_loss["exp_decay_weight"]
            )
            self.logger.record("train/criticnet_loss", criticnet_loss["loss"])
            self.logger.record("train/valuenet_loss", valuenet_loss["loss"])
            self.logger.record("train/actornet_loss", actornet_loss["loss"])
            self.logger.record("train/q_minus_v_mean", valuenet_loss["q_minus_v_mean"])
            self.logger.record("train/mean_reward", criticnet_loss["mean_reward"])
            self.logger.dump(step=self.num_timesteps)

    def _exponential_decay(
        self, features1: torch.Tensor, features2: torch.Tensor, std: float = 1.0
    ):
        """
        Exponential difference between two features.
        Args:
            features1: (torch.Tensor) Features 1. (B, D)
            features2: (torch.Tensor) Features 2. (B, D)
            std: (float) Exponential scaling factor.
        """
        diff = features1 - features2
        norm = torch.linalg.norm(diff, dim=1, keepdim=True)
        return torch.exp(-norm / std**2)

    def _update_visitnet(
        self,
        state,
        action,
        done,
        next_state,
        next_action=None,
        perturbed_state=None,
        perturbed_action=None,
    ):
        self.visitnet.set_training_mode(True)
        # --- Bellman Loss ---
        current_sr = self.visitnet(state, action)
        current_features = self.visitnet.extract_features(
            state, self.visitnet.features_extractor
        )
        with torch.no_grad():
            current_features_actions = torch.cat(
                [current_features.detach(), action], dim=1
            )
            next_sr = self.visitnet(next_state, next_action)
            target = current_features_actions + (1 - done) * self.gamma * next_sr
        bellman_loss = F.mse_loss(current_sr, target)

        # --- Reconstruction Loss ---
        next_features = self.visitnet.extract_features(
            next_state, self.visitnet.features_extractor
        ).detach()
        predicted_next_features = self.visitnet.decode(
            torch.cat([current_features, action], dim=1)
        )
        reconstruction_loss = F.mse_loss(predicted_next_features, next_features)

        # --- Norm Loss ---
        norm_loss = torch.pow(
            F.relu(
                torch.norm(
                    self.visitnet.sr_net(current_features_actions),
                    dim=1,
                    keepdim=True,
                    p=2,
                )
                - 1.0
            ),
            2,
        ).mean()

        if (perturbed_state is not None) and (perturbed_action is not None):
            # --- Negative sampling ---
            # 1) extract features for perturbed and unperturbed states
            # 2) compute the exponential decay weights only based on features
            # 3) compute SR(s_p, a_p) (with gradients)
            # 4) compute perturbed reward r_p = sr_to_reward(SR(s_p, a_p).detach())
            # 5) compute target reward r = sr_to_reward(SR(s, a)) * exp_decay_weight
            # 6) compute ns_loss: F.mse_loss(r_p, r)

            with torch.no_grad():
                perturbed_features = self.visitnet.extract_features(
                    perturbed_state, self.visitnet.features_extractor
                )
                exp_decay_weight = self._exponential_decay(
                    torch.cat([current_features.detach(), action], dim=1),
                    torch.cat([perturbed_features, perturbed_action], dim=1),
                    std=self.ns_noise_penalty_std,
                )
            perturbed_sr = self.visitnet(perturbed_state, perturbed_action)
            ns_loss = F.mse_loss(perturbed_sr, current_sr.detach() * exp_decay_weight)

        else:
            ns_loss = torch.Tensor([0.0]).to(bellman_loss.device)
            exp_decay_weight = torch.Tensor([0.0])

        loss = (
            bellman_loss
            + self.reconstruction_loss_coef * reconstruction_loss
            + self.norm_loss_coef * norm_loss
            + self.neg_sampling_loss_coef * ns_loss
        )
        self.visitnet.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.visitnet.parameters(), 10.0)
        self.visitnet.optimizer.step()

        self.visitnet.set_training_mode(False)
        return {
            "loss": loss.item(),
            "bellman_loss": bellman_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "norm_loss": norm_loss.item(),
            "ns_loss": ns_loss.item(),
            "exp_decay_weight": exp_decay_weight.mean().item(),
        }

    def _update_value_network(
        self,
        state,
        action,
    ):
        with torch.no_grad():
            q_s_a, _ = torch.min(
                torch.concat(self.critic_target(state, action), dim=1),
                dim=1,
                keepdim=True,
            )  # use the critic

        self.valuenet.set_training_mode(True)  # training mode ON
        v_s = self.valuenet(state)
        diff = q_s_a - v_s

        if self.value_loss_fn == "chi-square":
            loss = self._chi_square_loss(diff, self.lambda_)

        elif self.value_loss_fn == "total-variation":
            loss = self._total_variation_loss(diff, self.lambda_)

        elif self.value_loss_fn == "reverse-kl":
            loss = self._reverse_kl_loss(diff, self.lambda_, self.max_clip)

        loss = loss.mean()

        self.valuenet.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.valuenet.parameters(), 10.0)
        self.valuenet.optimizer.step()
        self.valuenet.set_training_mode(False)  # training mode OFF
        return {
            "loss": loss.item(),
            "q_minus_v_mean": (q_s_a - v_s).mean().item(),
        }

    def _update_critic_network(
        self,
        state,
        next_state,
        action,
        done,
        reward=None,
    ):
        with torch.no_grad():
            reward = (
                reward.reshape(-1, 1)
                if self.use_ground_truth_reward
                else sr_to_reward(self.visitnet(state, action)).reshape(-1, 1)
            )
            v_s_prime = self.valuenet(next_state)
            q_target = reward + (1 - done) * self.gamma * v_s_prime

        self.critic.set_training_mode(True)  # training mode ON
        q_pred = torch.cat(self.critic(state, action), dim=1)
        q_target = torch.tile(q_target, (1, q_pred.shape[1]))
        loss = F.mse_loss(q_pred, q_target)

        self.critic.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 10.0)
        self.critic.optimizer.step()
        self.critic.set_training_mode(False)  # training mode OFF
        # update target networks
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        return {
            "loss": loss.item(),
            "mean_reward": reward.mean().item(),
        }

    def _update_actor_network(
        self,
        state,
        action,
    ):
        """
        fDVL [1] Actor loss:
        loss (forward KL) = -(exp((Q(s,a) - V(s)) * lambda_)) * log pi_actor(a|s))
        loss (reverse KL) = Q(s,a) - log(pi_actor(a|s)/mu(a|s))
        """
        self.actor.set_training_mode(True)  # training mode ON
        with torch.no_grad():
            q_s_a, _ = torch.min(
                torch.cat(self.critic_target(state, action), dim=1), dim=1, keepdim=True
            )
            v_s = self.valuenet(state)

            z = q_s_a - v_s
            exp_z = torch.clamp(torch.exp(z * self.alpha), max=100)

        logprob = self.actor.action_log_prob(state, action)
        logprob = logprob.reshape(-1, 1)
        loss = -(exp_z * logprob).mean()

        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        self.actor.set_training_mode(False)  # training mode OFF
        return {"loss": loss.item()}

    def _chi_square_loss(self, diff, lambda_):
        """
        Pearson chi-square loss: (lambda_ * max(diff + (diff**2) / 4, 0) - (1 - lambda_) * diff)
        """
        loss = (
            lambda_ * torch.maximum(diff + (diff**2) / 4, torch.tensor(0.0))
            - (1 - lambda_) * diff
        )
        return loss

    def _total_variation_loss(self, diff, lambda_):
        loss = lambda_ * torch.maximum(diff, torch.tensor(0.0)) - (1 - lambda_) * diff
        return loss

    def _reverse_kl_loss(
        self, diff, lambda_, max_clip=None
    ):  # COMMENT: same as gumbel_rescale_loss in XQL
        """
        Gumbel loss J: E[e^x - x - 1]. For stability to outliers, we scale the gradients with the max value over a batch
        and optionally clip the exponent. This has the effect of training with an adaptive lr.
        """
        z = diff / lambda_
        if max_clip is not None:
            z = torch.clamp(z, max=max_clip)
        max_z = torch.max(z, dim=0).values
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0), max_z)
        loss = torch.exp(z - max_z) - z * torch.exp(-max_z) - torch.exp(-max_z)
        return loss

    def _expectile_loss(diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff**2)
