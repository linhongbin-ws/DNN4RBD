import gym
import gym_acrobot
import time
from Controller import PD_Controller
from Trajectory import CosTraj
import matplotlib.pyplot as plt

env = gym.make('acrobotBmt-v0')
env.dt =0.01
controller = PD_Controller()
traj = CosTraj()
traj.A = 1


obsve = env.reset()
env.render()
time.sleep(2)
tCount = 0

t_list = []
q_des_list = []
q_list = []
for t in range(1500):
    tCount += env.dt
    env.render()
    q, qd, qdd = traj.forward(tCount)
    a = controller.forward(s=[obsve[0,0],obsve[1,0],obsve[2,0],obsve[3,0]], sDes=[q[0], q[1],qd[0],qd[1]])
    obsve, reward, done, info = env.step(a[0],a[1])
    t_list.append(tCount)
    q_des_list.append(q[0])
    q_list.append(obsve[0,0])
    #time.sleep(0.05)
env.close()

plt.figure()
plt.plot(t_list, q_des_list, 'bo', t_list, q_list, 'k')
plt.show()
