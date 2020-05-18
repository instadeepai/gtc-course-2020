import numpy as np
import flatland
from flatland.envs.rail_generators import complex_rail_generator

from environments.custom_schedule_generator import sparse_schedule_generator
from environments.custom_rail_generator import simple_rail_generator
from environments.env import FlatlandMultiAgentEnv
from environments.observations import TreeObsForRailEnv


def generator_from_seed(nr_start_goal=1, nr_extra=100, min_dist=20, max_dist=99999, seed=0):

    rail_generator = complex_rail_generator(nr_start_goal, nr_extra, min_dist, max_dist, seed)

    def generator(width, height, num_agents, num_resets=0):
        return rail_generator(width, height, num_agents, 0)

    return generator


def env_from_env_config(env_config):
    """
    flatland env from env_config

    - "rail_generator": function name of rail_generator in `observations.py`
    - "rail_config": kwargs for the rail_generator
    - "frozen": random seed frozen / changing
    """

    required_keys = [
        "obs_config",
        "rail_generator",
        "rail_config",
        "width",
        "height",
        "number_of_agents",
        "schedule_generator",
        "schedule_config",
        "frozen",
        "remove_agents_at_target",
        "wait_for_all_done"
    ]

    for k in required_keys:
        assert k in env_config, "Error: missing key: {} in env_config".format(k)

    if env_config["frozen"]:
        rail_generator = generator_from_seed
    else:
        rail_generator = getattr(flatland.envs.rail_generators, env_config["rail_generator"])
    schedule_generator = getattr(flatland.envs.schedule_generators, env_config["schedule_generator"])
    
    speed_ration_map = {1.: 1.}

    return FlatlandMultiAgentEnv(
        width=env_config["width"],
        height=env_config["height"],
        rail_generator=rail_generator(**env_config["rail_config"]),
        schedule_generator=schedule_generator(**env_config["schedule_config"]),
        number_of_agents=env_config["number_of_agents"],
        obs_builder_object=TreeObsForRailEnv(**env_config["obs_config"]),
        remove_agents_at_target=env_config["remove_agents_at_target"],
        wait_for_all_done=env_config["wait_for_all_done"],
        name=env_config.get("name", "flatland_env")
    )


def pad_done_agents(obs, rewards, dones, observation_space, n_trains):
    """
    For trains that have left the environment, pad:
        > missing observations with zeros
        > dones with `True`
        > rewards with 0.0
    at the correct indices for the missing trains
    """
    for i in range(n_trains):
        if i not in dones.keys():
            dones.update({i: True})
            obs = np.insert(obs, i , np.zeros(observation_space,))
    
    new_rewards = []
    for i in range(n_trains):
        if dones[i]:
            new_rewards.append(0.0)
        else:
            new_rewards.append(rewards[i])
            
    return obs, new_rewards, dones


def pad_initial_obs(obs, observation_space, n_trains):
    """
    For trains that have not yet entered the environment,
    pad missing observations with zeros.
    """
    diff = observation_space * n_trains - obs.shape[0]
    obs = np.pad(obs, (0, diff), mode="constant")
    return obs
