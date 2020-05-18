import time
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.schedule_generators import random_schedule_generator
from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv
from environments.observations import TreeObsForRailEnv


class FlatlandMultiAgentEnv(MultiAgentEnv):
    """
    Wrap a flatland RailEnv as an Rllib MultiAgentEnv.
    
    width, height, number_of_agents: int
    remove_agents_at_target: bool
    """

    def __init__(
        self,
        width,
        height,
        rail_generator,
        number_of_agents,
        remove_agents_at_target,
        obs_builder_object,
        wait_for_all_done,
        schedule_generator=random_schedule_generator(),
        name=None
    ):
        super().__init__()

        self.env = RailEnv(
            width=width,
            height=height,
            rail_generator=rail_generator,
            schedule_generator=schedule_generator,
            number_of_agents=number_of_agents,
            obs_builder_object=obs_builder_object,
            remove_agents_at_target=remove_agents_at_target,
        )

        self.wait_for_all_done = wait_for_all_done
        self.env_renderer = None
        self.agents_done = []
        self.frame_step = 0
        self.name = name
        self.number_of_agents = number_of_agents

        # Track when targets are reached. Ony used for correct reward propagation
        # when using wait_for_all_done=True
        self.at_target = dict(zip(list(np.arange(self.number_of_agents)),
                                [False for _ in range(self.number_of_agents)]))        

    def _running_agents(self):
        """
        Return IDs of the agents that are not done
        """
        agents = range(len(self.env.agents))
        return (i for i in agents if i not in self.agents_done)

    def _agents_not_at_target(self):
        """
        Return the number of agents that are not at their targets.
        Used when wait_for_all_done=True
        """
        return max(1, list(self.at_target.values()).count(False))

    def step(self, action_dict):
        """
        Env step for each agent, like a gym.step() call
        
        The action_dict object is a dict with str or int keys corresponding to agent IDs
        E.g: {'0': ..., '1': ..., ...} or {0: ..., 1: ..., ...}
        
        Return a dict with keys:
            "observations"
            "rewards"
            "dones"
            "infos"
        """
        obs, rewards, dones, infos = self.env.step(action_dict)
        o, r, d, i = {}, {}, {}, {}

        for agent in self._running_agents():
            o[agent] = obs[agent]
            r[agent] = rewards[agent] / self._agents_not_at_target()
            i[agent] = infos

            if self.wait_for_all_done:
                dones, r, i = self._process_all_done(agent, dones, r, i)

            d[agent] = dones[agent]
            
        d["__all__"] = dones["__all__"]

        for agent, done in dones.items():
            if agent != "__all__" and done:
                self.agents_done.append(agent)

        self.frame_step += 1

        return o, r, d, i

    def reset(self):
        """
        Return a dict {agent_id: agent_obs, ...}
        """
        self.agents_done = []
        obs, _ = self.env.reset()
        if self.env_renderer:
            self.env_renderer.set_new_rail()
        return obs

    def render(self, **kwargs):
        from flatland.utils.rendertools import RenderTool

        if not self.env_renderer:
            self.env_renderer = RenderTool(self.env, gl="PILSVG")
            self.env_renderer.set_new_rail()
        self.env_renderer.render_env(show=True, frames=False, show_observations=False, **kwargs)
        time.sleep(0.1)
        self.env_renderer.render_env(show=True, frames=False, show_observations=False, **kwargs)
        return self.env_renderer.get_image()

    def _process_all_done(self, agent, dones, r, i):
        # Do not count target reward more than once
        if self.at_target[agent]:
            r[agent] = 0.0
        
        # If agent is done, and the group is not done, and agent has
        # not previously reached the target 
        if dones[agent] and not dones['__all__']:
            self.at_target[agent] = True
            
        # Ensure each individual agent is only marked 'done' when all are done
        for a in list(dones.keys()):
            dones[a] = dones['__all__']
        
        return dones, r, i

    @property
    def action_space(self):
        return Discrete(5)

    @property
    def observation_space(self):
        size, pow4 = 0, 1
        for _ in range(self.env.obs_builder.max_depth + 1):
            size += pow4
            pow4 *= 4
        observation_size = size * self.env.obs_builder.observation_dim
        return Box(-np.inf, np.inf, shape=(observation_size,))
