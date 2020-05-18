from typing import Tuple, List, Callable, Mapping, Optional, Any

import msgpack
import sys
import numpy as np
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.schedule_utils import Schedule

AgentPosition = Tuple[int, int]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Schedule]



def sparse_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:
    """

    This is the schedule generator which is used for Round 2 of the Flatland challenge. It produces schedules
    to railway networks provided by sparse_rail_generator.
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = np.random) -> Schedule:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the schedule
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """

        _runtime_seed = seed + num_resets

        train_stations = hints['train_stations']
        city_positions = hints['city_positions']
        city_orientation = hints['city_orientations']
        max_num_agents = hints['num_agents']
        city_orientations = hints['city_orientations']
        if num_agents > max_num_agents:
            num_agents = max_num_agents
            warnings.warn("Too many agents! Changes number of agents.")
        # Place agents and targets within available train stations
        agents_position = []
        agents_target = []
        agents_direction = []
        taken_starts = set()
        taken_targets = set()
        start_city_station = None
        target_city_station = None
        available_target = set(list(np.arange(num_agents)))
        for agent_idx in range(num_agents):
            # import pdb; pdb.set_trace()
            targets = list(available_target-set((agent_idx,)))
            if not targets:
                targets = [agent_idx]
            target_idx = np_random.choice(targets, 1)[0]
            taken_starts.add(agent_idx)
            taken_targets.add(target_idx)
            available_target = available_target-taken_targets
            taken_starts.add(agent_idx)
            start_city_stations = train_stations[agent_idx]
            target_city_stations = train_stations[target_idx]
            # get start/target stations at cities
            for station in start_city_stations:
                if station[1] % 2 == 0:
                    start_city_station = station
                    break
            for station in target_city_stations:
                if station[1] % 2 == 1:
                    target_city_station = station
                    break
            if start_city_station is None or target_city_station is None:
                sys.exit("Could not schedule agents. Invalid parameters")
            agent_orientation = city_orientation[agent_idx]
            agents_position.append(start_city_station[0])
            agents_target.append(target_city_station[0])
            agents_direction.append(agent_orientation)

        if speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)

        # We add multiply factors to the max number of time steps to simplify task in Flatland challenge.
        # These factors might change in the future.
        timedelay_factor = 4
        alpha = 2
        max_episode_steps = int(
            timedelay_factor * alpha * (rail.width + rail.height + num_agents / len(city_positions)))
        # import pdb; pdb.set_trace()
        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None)
                        # max_episode_steps=max_episode_steps)

    return generator


def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None,
                                seed: int = None, np_random: RandomState = None) -> List[float]:
    """
    Parameters
    ----------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
    List[float]
        A list of size nb_agents of speeds with the corresponding probabilistic ratios.
    """
    if speed_ratio_map is None:
        return [1.0] * nb_agents

    nb_classes = len(speed_ratio_map.keys())
    speed_ratio_map_as_list: List[Tuple[float, float]] = list(speed_ratio_map.items())
    speed_ratios = list(map(lambda t: t[1], speed_ratio_map_as_list))
    speeds = list(map(lambda t: t[0], speed_ratio_map_as_list))
    return list(map(lambda index: speeds[index], np_random.choice(nb_classes, nb_agents, p=speed_ratios)))