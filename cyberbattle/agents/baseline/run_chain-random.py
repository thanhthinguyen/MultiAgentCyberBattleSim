import sys
import logging
import gym
from gym import spaces
import numpy as np
import networkx as nx
import cyberbattle.simulation.actions as actions
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import cyberbattle.agents.random_agent as random_agent
import cyberbattle.samples.chainpattern.chainpattern as chainpattern
import importlib
importlib.reload(actions)
importlib.reload(cyberbattle_env)
importlib.reload(chainpattern)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")

gym_env = gym.make('CyberBattleChain-v0', size=10, attacker_goal=None)
gym_env.environment
gym_env.environment.network.nodes
gym_env.action_space
gym_env.action_space.sample()
gym_env.observation_space.sample()

random_agent.run_random_agent(1, 10000, gym_env)

#for iter in range(100):
    # gym_env.sample_valid_action()
    # o,r,d,i = gym_env.step(gym_env.sample_valid_action())
    # o
