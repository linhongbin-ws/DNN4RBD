import gym
import gym_acrobot
import time

env = gym.make('acrobotBmt-v0')

for i_episode in range(1):
    observation = env.reset()
    env.render()
    time.sleep(2)
    for t in range(100):
        env.render()
        print(observation)
        #action = env.action_space.sample()
        action = 1
        observation, reward, done, info = env.step(0,0)
        time.sleep(0.05)

env.close()