import gym
import gym_acrobot
import time
from Controller import PD_Controller
from Trajectory import CosTraj
import numpy

env = gym.make('acrobotBmt-v0')
env.dt =0.01
controller = PD_Controller()


obsve = env.reset()
env.render()
time.sleep(2)
tCount = 0
for t in range(3000):
    tCount += env.dt
    env.render()
    qDes = [numpy.pi,0]
    a = controller.forward(s=[obsve[0,0],obsve[1,0],obsve[2,0],obsve[3,0]], qDes=qDes)
    obsve, reward, done, info = env.step(a[0],a[1])
    print(obsve)
    #time.sleep(0.05)