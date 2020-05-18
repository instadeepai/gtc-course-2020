#!/usr/bin/env python

import collections
import gym
import os
import pickle
import ray
from pathlib import Path
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import register_env

from models.centralized_critic_model import CentralizedCriticModel
from models.dense_model import DenseModel
from environments.env_utils import env_from_env_config
from environments.preprocessor import TreeObsPreprocessor
from gym.wrappers.monitor import Monitor


register_env("train_env", env_from_env_config)
ModelCatalog.register_custom_model("dense_model", DenseModel)
ModelCatalog.register_custom_model("cc_dense", CentralizedCriticModel)
ModelCatalog.register_custom_preprocessor("tree_obs_preprocessor", TreeObsPreprocessor)


def run_rollout(run_algo, checkpoint, params_path, env, steps=60, episodes=0, no_render=False, monitor=False):
    """Note: this assume ray.init() has been called."""
    config = {}

    # Load configuration from file
    config_dir = os.path.dirname(params_path)
    config_path = os.path.join(config_dir, "params.pkl")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])

    cls = get_agent_class(run_algo)
    agent = cls(env=env, config=config)
    agent.restore(checkpoint)
    num_steps = int(steps)
    num_episodes = int(episodes)

    return rollout(agent, env, num_steps, num_episodes, no_render, monitor)


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            no_render=True,
            monitor=False):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: _flatten_action(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    frames = []
    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    a_action = agent.compute_action(
                        a_obs,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                    a_action = _flatten_action(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                frames.append(env.render())
            steps += 1
            obs = next_obs
        print(f"Episode #{episodes}: reward: {reward_total}")
        if done:
            episodes += 1
        
        return frames
