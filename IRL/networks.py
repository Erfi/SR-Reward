from typing import List, Type, Tuple, Dict
import math

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from stable_baselines3.common.policies import BaseModel
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim

from IRL.utils import init_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        half_d = d // 2
        log_denominator = -math.log(n) / (half_d - 1)
        denominator_ = torch.exp(torch.arange(half_d) * log_denominator)
        self.register_buffer("denominator", denominator_)

    def forward(self, time):
        """
        :param time: shape=(B, )
        :return: Positional Encoding shape=(B, d_model)
        """
        argument = time[:, None] * self.denominator[None, :]  # (B, half_d_model)
        return torch.cat([argument.sin(), argument.cos()], dim=-1)  # (B, d_model)


class IRLActor(Actor):
    def __init__(
        self, *args, log_std_max: float = 2, log_std_min: float = -20, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

        # init weights
        self.mu.apply(init_weights)
        self.latent_pi.apply(init_weights)

    def get_action_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_actions, log_std, {}

    def action_log_prob(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log probability of the action under the current distribution.
        If no action is given, then actions are sampled from the distribution, and their log probability is returned.
        Otherwise, the log probability of the given action is returned.
        """
        if action is None:
            return super().action_log_prob(obs)
        else:
            # current distribution parameters
            cur_mean, cur_log_td, _ = self.get_action_dist_params(obs)
            # update current action distribution
            self.action_dist.proba_distribution(
                mean_actions=cur_mean, log_std=cur_log_td
            )
            # get the log prob of the given action under the current distribution
            return self.action_dist.log_prob(action)


class SRNet(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        decoder_net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.Mish,
        normalize_images: bool = True,
        share_features_extractor: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        sr_net = create_mlp(
            features_dim + action_dim,
            features_dim + action_dim,
            net_arch,
            activation_fn,
            squash_output=False,
            use_layer_norm=use_layer_norm,
        )
        self.sr_net = nn.Sequential(*sr_net)
        decode_net = create_mlp(
            input_dim=features_dim + action_dim,
            output_dim=features_dim,
            # output_dim=observation_space.shape[-1],
            net_arch=decoder_net_arch,
            activation_fn=activation_fn,
            squash_output=False,
            use_layer_norm=use_layer_norm,
        )
        self.decode_net = nn.Sequential(*decode_net)

        # Initialize weights
        self.sr_net.apply(init_weights)
        self.decode_net.apply(init_weights)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        sr_input = torch.cat([features, actions], dim=1)
        if return_features:
            return self.sr_net(sr_input), features
        return self.sr_net(sr_input)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature back to the state (reconstruction).
        """
        return self.decode_net(x)


class ValueNet(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.Mish,
        normalize_images: bool = True,
        share_features_extractor: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.share_features_extractor = share_features_extractor
        v_net = create_mlp(
            features_dim,
            1,
            net_arch,
            activation_fn,
            squash_output=False,
            use_layer_norm=use_layer_norm,
        )
        self.v_net = nn.Sequential(*v_net)

        # Initialize weights
        self.v_net.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return self.v_net(features)


class ContinuousCritic(BaseModel):
    """
    Taken from Stable Baselines 3, but
    Modified to accept LayerNorm
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(
                features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
                use_layer_norm=use_layer_norm,
            )
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        # Initialize weights
        for q_net in self.q_networks:
            q_net.apply(init_weights)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](torch.cat([features, actions], dim=1))


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    use_layer_norm: bool = False,
) -> List[nn.Module]:
    """
    NOTE: borrowed from Stable Baselines 3, but modified to accept layer norm
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output vector
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias)]
        if use_layer_norm:
            modules.append(nn.LayerNorm(net_arch[0]))
        modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        if use_layer_norm:
            modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# ==================================================== #
# ==== LSTM and MLP and LastN Features Extractors ==== #
# ==================================================== #


class BaseSeqFeaturesExtractor(BaseFeaturesExtractor):
    """
    Base class for sequence feature extractor
    The output of this feature extractor is a sequence of features
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LSTM_Output_Extractor(nn.Module):
    """
    Convenient Class to extract the last output of a LSTM
    """

    def forward(self, x):
        # Output shape of LSTM with batch_first=True: (batch, seq, hidden), (h, c)
        tensor, _ = x
        # Return last output of sequence (batch, hidden)
        return tensor[:, -1, :]


class LSTMFeaturesExtractor(BaseSeqFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        hidden_size: int,
        num_layers: int,
        relu_finish: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        :param observation_space: Observation space of the environment
        :param hidden_size: hidden size of the LSTM (this will be the features_dim that is outputted)
        :param num_lstm_layers: number of LSTM layers
        :param relu_finish: whether to apply ReLU activation at the end
        :param normalize: whether to normalize the output
        """
        super().__init__(observation_space, features_dim=hidden_size)
        self.relu_finish = relu_finish
        self.normalize = normalize
        # input features are after the sequence dimension
        input_size = np.prod(observation_space.shape[1:])
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lstm_output_extractor = LSTM_Output_Extractor()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: (batch, seq, features)
        """
        x = self.flatten(observations)
        x = self.lstm(x)
        x = self.lstm_output_extractor(x)
        if self.relu_finish:
            x = F.relu(x)
        if self.normalize:
            x = x / (torch.norm(x, p=1, dim=-1, keepdim=True) + 1e-8)
        return x


class FlattenLastNExtractor(BaseSeqFeaturesExtractor):
    """
    Feature extractor that flatten the last element of the input sequence.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space: Observation space of the environment (has sequence length)
    NOTE: we flatten everything AFTER batch and only for the last N element of the sequence.
    NOTE: all elements before the last N element are ignored
    """

    def __init__(self, observation_space: spaces.Space, last_n: int, skip: int) -> None:
        """
        We can concatenate the last n elements of the sequence, but skip some of them.
        :param observation_space: Observation space of the environment
        :param last_n: number of last observations of the sequence to consider for concatenation
        :param skip: number of observations to skip between each observation

        resulting flatten(onservation[:, -last_n::skip, :]) resulting in (last_n - 1)//skip + 1 observations
        NOTE: In order to include the ltest observation choose last_n and skip such that (last_n - 1) % skip == 0
        """
        self.last_n = last_n
        self.skip = skip
        n_obs = (last_n - 1) // skip + 1
        super().__init__(
            observation_space, np.prod(observation_space.shape[1:]) * n_obs
        )
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Flatten and output the last observation in the sequence
        """
        return self.flatten(observations[:, -self.last_n :: self.skip, :])


class MLPFlattenLastNExtractor(BaseSeqFeaturesExtractor):
    """
    Feature extractor that passes the last N observations through an MLP.
    :param observation_space: Observation space of the environment (has sequence length)
    NOTE: we flatten everything AFTER batch and only for the last N element of the sequence.
    NOTE: all elements before the last N element are ignored
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        net_arch: List[int],
        output_dim: int,
        last_n: int,
        skip: int,
        relu_finish: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        We can concatenate the last n elements of the sequence, but skip some of them.
        :param observation_space: Observation space of the environment
        :param net_arch: architecture of the MLP
        :param last_n: number of last observations of the sequence to consider for concatenation
        :param skip: number of observations to skip between each observation

        resulting flatten(onservation[:, -last_n::skip, :]) resulting in (last_n - 1)//skip + 1 observations
        NOTE: In order to include the ltest observation choose last_n and skip such that (last_n - 1) % skip == 0
        """
        self.relu_finish = relu_finish
        self.normalize = normalize
        self.last_n = last_n
        self.skip = skip
        n_obs = (last_n - 1) // skip + 1
        super().__init__(observation_space, output_dim)
        self.flatten = nn.Flatten(start_dim=1)
        mlp = create_mlp(
            input_dim=np.prod(observation_space.shape[1:]) * n_obs,
            output_dim=output_dim,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            squash_output=False,
            use_layer_norm=False,
        )
        self.mlp = nn.Sequential(*mlp)

        # init weights
        self.mlp.apply(init_weights)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.flatten(observations[:, -self.last_n :: self.skip, :])
        x = self.mlp(x)
        if self.relu_finish:
            x = F.relu(x)
        if self.normalize:
            x = x / (torch.norm(x, p=1, dim=-1, keepdim=True) + 1e-8)
        return x


class NavMapExtractor(BaseSeqFeaturesExtractor):
    """
    Passes the last flattened observations through a position encoder and an MLP
    This is totally specific to NavMap. please DO NOT USE GENERALLY
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        hidden_size: int,
        use_pos_enc: bool,
        pos_enc_dim: int,
        pos_enc_n: int,
    ) -> None:
        super().__init__(observation_space, hidden_size)
        self.flatten = nn.Flatten(start_dim=1)
        self.use_pos_enc = use_pos_enc
        self.positional_encoding = PositionalEncoding(d=pos_enc_dim, n=pos_enc_n)
        flatten_dim = 2 * pos_enc_dim if use_pos_enc else 2
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Flatten and output the last observation in the sequence
        """
        x = self.flatten(observations[:, -1, :])
        if self.use_pos_enc:
            assert x.shape[1] == 2, f"Expected shape (B, 2), got {x.shape}"
            x_encode = self.positional_encoding(x[:, 0])
            y_encode = self.positional_encoding(x[:, 1])
            x = torch.cat([x_encode, y_encode], dim=1)
        x = self.mlp(x)
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x
