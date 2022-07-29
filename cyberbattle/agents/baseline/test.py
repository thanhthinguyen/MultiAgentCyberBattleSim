import gym
import cyberbattle._env.cyberbattle_env as cyberbattle_env

cyberbattlechain = gym.make('CyberBattleChain-v0',
                            size=10,
                            attacker_goal=cyberbattle_env.AttackerGoal(
                                own_atleast_percent=1,
                                reward=1000))

cyberbattlechain.environment.plot_environment_graph()

print()

