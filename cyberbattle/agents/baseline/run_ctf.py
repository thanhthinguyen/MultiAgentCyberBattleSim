import sys
import torch
import logging
import asciichartpy
import gym
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_randomcredlookup as rca
import cyberbattle.agents.baseline.agent_tabularqlearning as tqa
import cyberbattle.agents.baseline.agent_dql as dqla
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

import cyberbattle._env.cyberbattle_env


#  logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")
logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %% {"tags": ["parameters"]}
# Papermill notebook parameters


print(f"torch cuda available={torch.cuda.is_available()}")

#############
# gymid = 'CyberBattleTiny-v0'
#############
gymid = "CyberBattleToyCtf-v0"
env_size = None
iteration_count = 1000  # 1500
training_episode_count = 10  # 10  # 20
eval_episode_count = 5  # 10
maximum_node_count = 12
maximum_total_credentials = 10
#############
# gymid = "CyberBattleChain-v0"
# env_size = 10
# iteration_count = 9000
# training_episode_count = 50
# eval_episode_count = 5
# maximum_node_count = 22
# maximum_total_credentials = 22

# %%
# Load the Gym environment
if env_size:
    gym_env = gym.make(gymid, size=env_size)
else:
    gym_env = gym.make(gymid)

ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=maximum_node_count,
    maximum_total_credentials=maximum_total_credentials,
    identifiers=gym_env.identifiers
)

# %%
debugging = True
if debugging:
    print(f"port_count = {ep.port_count}, property_count = {ep.property_count}")

    gym_env.environment
    # training_env.environment.plot_environment_graph()
    gym_env.environment.network.nodes
    gym_env.action_space
    gym_env.action_space.sample()
    gym_env.observation_space.sample()
    o0 = gym_env.reset()
    o_test, r, d, i = gym_env.step(gym_env.sample_valid_action())
    o0 = gym_env.reset()

    o0.keys()

    fe_example = w.RavelEncoding(ep, [w.Feature_active_node_properties(ep), w.Feature_discovered_node_count(ep)])
    a = w.StateAugmentation(o0)
    w.Feature_discovered_ports(ep).get(a, None)
    fe_example.encode_at(a, 0)

# %%
all_runs = []

print("Evaluate the DQL agent")
learner0 = dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    )
dql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=learner0,
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,  #Verbosity.Quiet
    render=True,
    plot_episodes_length=False,
    title="DQL"
)
all_runs.append(dql_run)

print("Evaluate the Random agent")
random_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=learner.RandomPolicy(),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=1.0,  # purely random
    render=True,
    verbosity=Verbosity.Quiet,
    title="Random search"
)
all_runs.append(random_run)

print("Evaluate the Multiagent DQL agent")
learner1 = dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    )
learner2 = dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    )
learners = []
learners.append(learner1)
learners.append(learner2)
dql_run_multi = learner.epsilon_greedy_search_multi(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learners=learners,
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,  #Verbosity.Quiet
    render=True,
    plot_episodes_length=False,
    title="DQL"
)
all_runs.append(dql_run_multi)

colors = [asciichartpy.red, asciichartpy.green, asciichartpy.yellow, asciichartpy.blue]

print("Episode duration -- DQN=Red, Random=Green, Multiagent DQN=Yellow")
print(asciichartpy.plot(p.episodes_lengths_for_all_runs(all_runs), {'height': 30, 'colors': colors}))

print("Cumulative rewards -- DQN=Red, Random=Green, Multiagent DQN=Yellow")
c = p.averaged_cummulative_rewards(all_runs, 80)  # args.rewardplot_width
print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))
