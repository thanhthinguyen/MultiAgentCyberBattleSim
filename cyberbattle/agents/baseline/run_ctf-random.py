import sys
import logging
import gym
import cyberbattle._env.cyberbattle_env


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
gym_env = gym.make('CyberBattleToyCtf-v0')

print(gym_env.environment)

print(gym_env.action_space)

print(gym_env.action_space.sample())

print("---------*************************---------")

# for i_episode in range(1):
#     observation = gym_env.reset()
#
#     total_reward = 0
#
#     for t in range(5600):  #5600
#         actions = []
#         action = gym_env.sample_valid_action()
#         actions.append(action)
#         action = gym_env.sample_valid_action()
#         actions.append(action)
#
#         print("--------thi: ", actions)
#         observations, rewards, dones, infos = gym_env.step(actions)
#
#         total_reward += rewards[0]
#         total_reward += rewards[1]
#
#         if rewards[0] > 0:
#             print('Agent 1 ####### rewarded action: {action}')
#             print(f'total_reward={total_reward} reward1={rewards[0]}')
#             gym_env.render()
#
#         if rewards[1] > 0:
#             print('Agent 2 ####### rewarded action: {action}')
#             print(f'total_reward={total_reward} reward2={rewards[1]}')
#             gym_env.render()
#
#         # if done:
#         #     print("Episode finished after {} timesteps".format(t + 1))
#         #     break
#
#     gym_env.render()
#
# gym_env.close()
# print("simulation ended")



for i_episode in range(1):
    observation = gym_env.reset()

    total_reward = 0

    for t in range(5600):
        action = gym_env.sample_valid_action()

        observation, reward, done, info = gym_env.step(action)

        total_reward += reward

        if reward > 0:
            print('####### rewarded action: {action}')
            print(f'total_reward={total_reward} reward={reward}')
            gym_env.render()

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    gym_env.render()

gym_env.close()
print("simulation ended")
