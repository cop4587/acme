from absl.testing import absltest

import numpy as np
import haiku as hk
from acme import specs
from acme.agents.jax import dqn
from acme.agents.jax.dqn import losses
from acme.utils import lp_utils
from acme.jax import experiments
from acme.jax import networks
from acme.jax import utils
from acme.jax import networks as networks_lib
import launchpad as lp

from acme.testing import fakes

SEED = 42
NUM_STEPS = 100


def make_dqn_builder():
  config = dqn.DQNConfig(
    discount=0.99,
    learning_rate=5e-5,
    n_step=1,
    epsilon=0.01,
    target_update_period=2000,
    min_replay_size=2_000,
    max_replay_size=100_000,
    samples_per_insert=8,
    batch_size=32)
  loss_fn = losses.PrioritizedDoubleQLearning(discount=config.discount)
  return dqn.DQNBuilder(config, loss_fn=loss_fn)


def make_config(builder, environment, network):
  return experiments.Config(
    builder=builder,
    environment_factory=lambda seed: environment,
    network_factory=lambda spec: network,
    policy_network_factory=dqn.behavior_policy,
    evaluator_factories=[],
    seed=SEED,
    max_number_of_steps=NUM_STEPS)


def make_discrete_network(
    spec: specs.EnvironmentSpec) -> networks.FeedForwardNetwork:
  def network(x):
    model = hk.Sequential([
      hk.Flatten(),
      hk.nets.MLP([50, 50, spec.actions.num_values]),
    ])
    return model(x)

  network_hk = hk.without_apply_rng(hk.transform(network))
  obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
  return networks_lib.FeedForwardNetwork(
    init=lambda rng: network_hk.init(rng, obs),
    apply=network_hk.apply)


def make_continuous_network(
    spec: specs.EnvironmentSpec) -> networks.FeedForwardNetwork:
  action_spec = spec.actions

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  def network(x):
    model = hk.Sequential([
      hk.Flatten(),
      hk.nets.MLP([50, 50, num_dimensions]),
      networks_lib.TanhToSpec(spec.actions),
    ])
    return model(x)

  network_hk = hk.without_apply_rng(hk.transform(network))
  obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
  return networks_lib.FeedForwardNetwork(
    init=lambda rng: network_hk.init(rng, obs),
    apply=network_hk.apply)


class ActorTest(absltest.TestCase):

  def test_behavior_policy_return_action_values(self):
    builder = make_dqn_builder()
    environment = fakes.DiscreteEnvironment(
      num_actions=5,
      num_observations=1,
      obs_shape=(1024,),
      obs_dtype=np.float32,
      episode_length=10)
    spec = specs.make_environment_spec(environment)
    network = make_discrete_network(spec)
    config = make_config(builder, environment, network)
    experiments.run_experiment(experiment=config)

  def test_jax_argmax_haiku_network_output(self):
    # Make actor_core policy.network ouputs Q-value for one action,
    # spike actor jax.argmax(batch_dim action_values)
    pass


if __name__ == '__main__':
  absltest.main()
