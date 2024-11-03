from typing import List, Tuple, Optional, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs

from dynsyn.algorithms.common.dynsynLayer import DynSynPPOLayer
from dynsyn.algorithms.common.utils import get_dynsyn_weight_amp

SelfDynSynPPO = TypeVar("SelfDynSynPPO", bound="DynSynPPO")


class DynSynActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        dynsyn: List[List[int]],
        dynsyn_log_std: float,
        **kwargs
    ) -> None:
        super().__init__(
            observation_space=observation_space, action_space=action_space, lr_schedule=lr_schedule, **kwargs
        )

        self.dynsyn = dynsyn
        self.dynsyn_log_std = dynsyn_log_std
        self.dynsyn_layer = DynSynPPOLayer(
            dynsyn, last_layer_dim=self.net_arch["pi"][-1], dynsyn_log_std=dynsyn_log_std
        )

        # Change the action dist dimension
        # TODO: Only support Box action space
        assert isinstance(action_space, spaces.Box), "Only Box action space is supported for now."
        self.modified_action_shape = (self.dynsyn_layer.muscle_group_num,)
        modified_action_space = spaces.Box(-np.inf, np.inf, shape=self.modified_action_shape)

        self.action_dist = make_proba_distribution(
            modified_action_space, use_sde=self.use_sde, dist_kwargs=self.dist_kwargs
        )
        self._build(lr_schedule=lr_schedule)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.modified_action_shape))  # type: ignore[misc]

        actions = self.dynsyn_layer(actions, latent_pi, deterministic)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        actions = self.dynsyn_layer.revert(actions)
        return super().evaluate_actions(obs, actions)


class DynSynPPO(PPO):
    def __init__(
        self,
        *args,
        dynsyn_k: float = 0,
        dynsyn_a: float = 0,
        dynsyn_weight_amp: Optional[float] = None,
        **kwargs
    ):
        # These parameters need to assign before the super().__init__() due to the _setup_model() function
        self.dynsyn_k = dynsyn_k
        self.dynsyn_a = dynsyn_a
        self.dynsyn_weight_amp = dynsyn_weight_amp

        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy.dynsyn_layer.update_dynsyn_weight_amp(self.dynsyn_weight_amp)

    def learn(self, *args, **kwargs) -> SelfDynSynPPO:
        dynsyn_weight_amp = get_dynsyn_weight_amp(0, self.dynsyn_k, self.dynsyn_a, self.dynsyn_weight_amp)
        self.policy.dynsyn_layer.update_dynsyn_weight_amp(dynsyn_weight_amp)
        return super().learn(*args, **kwargs)

    def train(self) -> None:
        dynsyn_weight_amp = get_dynsyn_weight_amp(
            self.num_timesteps, self.dynsyn_k, self.dynsyn_a, self.dynsyn_weight_amp
        )
        self.policy.dynsyn_layer.update_dynsyn_weight_amp(dynsyn_weight_amp)

        super().train()
        self.logger.record("train/dynsyn_weight_amp", dynsyn_weight_amp)
