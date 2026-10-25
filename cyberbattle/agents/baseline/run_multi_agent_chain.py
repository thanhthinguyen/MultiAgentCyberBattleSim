import torch
import gym
import logging
import sys
import asciichartpy
import cyberbattle._env.cyberbattle_env as cyberbattle_env
from cyberbattle._env.defender import ScanAndReimageCompromisedMachines
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.plotting as p
import numpy as np
import random
import cyberbattle.agents.baseline.learner as learner
import matplotlib.pyplot as plt
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

print(f"torch cuda available={torch.cuda.is_available()}")

# Completely reproducible results are not guaranteed across PyTorch releases,
# individual commits, or different platforms. Furthermore, results may not be
# reproducible between CPU and GPU executions, even when using identical seeds.
# https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed_):
    random.seed(seed_)
    os.environ['PYTHONHASHSEED'] = str(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)  # gpu vars
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)


def convert_stats(stats):
    list_ = [stats['explore']['reward']['local'], stats['explore']['noreward']['local'],
             stats['explore']['reward']['remote'], stats['explore']['noreward']['remote'],
             stats['explore']['reward']['connect'], stats['explore']['noreward']['connect'],
             stats['exploit']['reward']['local'], stats['exploit']['noreward']['local'],
             stats['exploit']['reward']['remote'], stats['exploit']['noreward']['remote'],
             stats['exploit']['reward']['connect'], stats['exploit']['noreward']['connect'],
             stats['exploit_deflected_to_explore']]
    return np.array(list_)


num_agents = 2
num_nodes = 10
num_episodes = 20
num_iterations = 500
# seed = 10
# set_seed(seed)
environment = 'CyberBattleChainMultiAgent-v0'


cc = []
com_stats = []
ind_stats = []
ran_stats = []
for i in range(20):
    cyberbattlechain = gym.make(environment,
                                num_agents=num_agents,
                                size=num_nodes,
                                attacker_goal=cyberbattle_env.AttackerGoal(
                                    own_atleast=0,
                                    own_atleast_percent=1.0
                                ), defender_constraint=cyberbattle_env.DefenderConstraint(
                                    maintain_sla=0.80
                                ),
                                defender_agent=ScanAndReimageCompromisedMachines(
                                    probability=0.1,
                                    scan_capacity=2,
                                    scan_frequency=5))

    ep = w.EnvironmentBounds.of_identifiers(
        maximum_total_credentials=33,
        maximum_node_count=33,
        maximum_discoverable_credentials_per_action=5,
        identifiers=cyberbattlechain.identifiers
    )

    all_runs = []
    manager_agent = dqla.ManagerDiscreteCommunicationModel(ep, num_agents)

    # Run Deep Q-learning with Communication Model
    dqn_learning_communication_run, stat = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain,
        environment_properties=ep,
        learners=[dqla.DeepQLearnerPolicy(
            agent_id=i,
            num_agents=num_agents,
            ep=ep,
            gamma=0.015,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.01,
            communication_model=dqla.CyberBattleStateActionModel(ep, i, num_agents),  # CyberBattleStateActionModel  CyberBattleStateActionCommunicationModel
            manager_agent=manager_agent
        ) for i in range(num_agents)],  # torch default is 1e-2
        episode_count=num_episodes,
        iteration_count=num_iterations,
        epsilon=0.90,
        render=False,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="Communication DQN"
    )
    all_runs.append(dqn_learning_communication_run)
    com_stats.append(convert_stats(stat))

    # Run Deep Q-learning with Independent Multi-Agent
    dqn_learning_run, stat = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain,
        environment_properties=ep,
        learners=[dqla.DeepQLearnerPolicy(
            agent_id=i,
            num_agents=num_agents,
            ep=ep,
            gamma=0.015,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.01) for i in range(num_agents)],  # torch default is 1e-2
        episode_count=num_episodes,
        iteration_count=num_iterations,
        epsilon=0.90,
        render=False,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="Independent DQN"
    )
    all_runs.append(dqn_learning_run)
    ind_stats.append(convert_stats(stat))

    random_run, stat = learner.epsilon_greedy_search(
        cyberbattlechain,
        ep,
        learners=[learner.RandomPolicy(agent_id=i) for i in range(num_agents)],
        episode_count=num_episodes,
        iteration_count=num_iterations,
        epsilon=1.0,  # purely random
        render=False,
        verbosity=Verbosity.Quiet,
        title="Random Search"
    )
    all_runs.append(random_run)
    ran_stats.append(convert_stats(stat))

    colors = [asciichartpy.red, asciichartpy.green, asciichartpy.yellow, asciichartpy.blue]

    print("Episode duration -- DQN-Communication=Red, DQN-Independence=Green, Random=Yellow")
    print(asciichartpy.plot(p.episodes_lengths_for_all_runs(all_runs), {'height': 20, 'colors': colors}))

    # print("Average rewards -- DQN-Communication=Red, DQN-Independence=Green, Random=Yellow")
    c = p.averaged_cummulative_rewards(all_runs, num_iterations)
    # print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))

    cc.append(c)
    d = np.average(np.array(cc), axis=0)
    ci = np.std(np.array(cc), axis=0)
    # print(d)
    print("Cumulative rewards -- DQN-Communication=Red, DQN-Independence=Green, Random=Yellow")
    print(asciichartpy.plot(d.tolist(), {'height': 20, 'colors': colors}))

    #========================================

    print("Episode duration -- DQN-Communication=Red, DQN-Independence=Green, Random=Yellow")
    e = p.episodes_lengths_for_all_runs(all_runs)
    print(asciichartpy.plot(e, {'height': 20, 'colors': colors}))

    plt.plot(list(range(0, len(e[0]))), e[0], label = "DQN-Communication", linestyle = "-")
    plt.plot(list(range(0, len(e[1]))), e[1], label = "DQN-Independence", linestyle = "-.")
    plt.plot(list(range(0, len(e[2]))), e[2], label = "Random Search", linestyle = "--")
    # plt.title("Episode duration")
    plt.xlabel("Episodes")
    plt.ylabel("Number of iterations")
    plt.legend(loc='upper left')
    plt.show()

    # print("Cumulative rewards -- DQN-Communication=Red, DQN-Independence=Green, Random=Yellow")
    # c = p.averaged_cummulative_rewards(all_runs, num_iterations)  # args.rewardplot_width  args.iteration_count
    # print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))

    # plt.yscale('log')
    # plt.xscale('log')
    plt.plot(list(range(0, len(d[0]))), d[0], label = "DQN-Communication", linestyle = "-")
    plt.fill_between(list(range(0, len(d[0]))), (d[0] - ci[0]), (d[0] + ci[0]), color='blue', alpha=0.1)
    plt.plot(list(range(0, len(d[1]))), d[1], label = "DQN-Independence", linestyle = "-.")
    plt.fill_between(list(range(0, len(d[1]))), (d[1] - ci[1]), (d[1] + ci[1]), color='orange', alpha=0.1)
    plt.plot(list(range(0, len(d[2]))), d[2], label = "Random Search", linestyle = "--")
    plt.fill_between(list(range(0, len(d[2]))), (d[2] - ci[2]), (d[2] + ci[2]), color='green', alpha=0.1)
    # plt.title("Cumulative rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Averaged cumulative rewards")
    plt.legend(loc='upper left')
    plt.show()


def dev_disp(x, y):
    sum_ = x + y
    if sum_ == 0:
        return 'NaN'
    else:
        return f"{(x / sum_):.2f}"


def print_table(x_stats):
    avrg = np.average(np.array(x_stats), axis=0)
    print(f"{avrg[0]:.1f}/{avrg[1]:.1f} "f"({dev_disp(avrg[0], avrg[1])})")
    print(f"{avrg[2]:.1f}/{avrg[3]:.1f} "f"({dev_disp(avrg[2], avrg[3])})")
    print(f"{avrg[4]:.1f}/{avrg[5]:.1f} "f"({dev_disp(avrg[4], avrg[5])})")
    print(f"{avrg[6]:.1f}/{avrg[7]:.1f} "f"({dev_disp(avrg[6], avrg[7])})")
    print(f"{avrg[8]:.1f}/{avrg[9]:.1f} "f"({dev_disp(avrg[8], avrg[9])})")
    print(f"{avrg[10]:.1f}/{avrg[11]:.1f} "f"({dev_disp(avrg[10], avrg[11])})")
    print(f"{avrg[12]:.1f}")


print_table(ran_stats)
print_table(ind_stats)
print_table(com_stats)
