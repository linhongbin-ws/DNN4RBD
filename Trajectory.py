import numpy as np
import gym
from gym import wrappers
import gym_acrobot
import time
from Controller import PD_Controller
import matplotlib.pyplot as plt
from os import path
class CosTraj():
    def __init__(self):
        self.A = np.pi
        self.w = np.array([1.0,1.0])
        self.b = np.array([0.,0.])
        self.dof = 2
    def forward(self, t):
        q = np.cos(self.w * t + self.b)*self.A
        qd = -self.w * np.sin(self.w * t + self.b)*self.A
        qdd = -self.w * self.w * np.cos(self.w * t + self.b) *self.A
        return q, qd, qdd


def genTrajectoryData(controller, traj, sampleNum = 20000, savePath='.',saveFig=True, dt=0.01):
    env = gym.make('acrobotBmt-v0')
    env.dt = dt
    obsve = env.reset()
    env.render()
    time.sleep(2)
    tCount = 0

    t_list = []
    q_des_dict= {}
    q_dict = {}
    qdot_dict = {}
    qddot_dict = {}
    a_dict = {}
    q_des_dict['J1'] = []
    q_des_dict['J2'] = []
    q_dict['J1'] = []
    q_dict['J2'] = []
    qdot_dict['J1'] = []
    qdot_dict['J2'] = []
    qddot_dict['J1'] = []
    qddot_dict['J2'] = []
    a_dict['J1'] = []
    a_dict['J2'] = []
    for t in range(sampleNum):
        tCount += env.dt
        env.render()
        q, qd, qdd = traj.forward(tCount)
        a = controller.forward(s=[obsve[0, 0], obsve[1, 0], obsve[2, 0], obsve[3, 0]], sDes=[q[0], q[1], qd[0], qd[1]])
        obsve, reward, done, info = env.step(a[0], a[1])
        t_list.append(tCount)
        q_des_dict['J1'].append(q[0])
        q_des_dict['J2'].append(q[1])
        q_dict['J1'].append(obsve[0, 0])
        q_dict['J2'].append(obsve[1, 0])
        qdot_dict['J1'].append(obsve[2, 0])
        qdot_dict['J2'].append(obsve[3, 0])
        a_dict['J1'].append(a[0])
        a_dict['J2'].append(a[1])
        if t==0:
            qddot_dict['J1'].append(0.)
            qddot_dict['J2'].append(0.)
        else:
            qddot_dict['J1'].append(qdot_dict['J1'][-1]-qdot_dict['J1'][-2])
            qddot_dict['J2'].append(qdot_dict['J2'][-1]-qdot_dict['J2'][-2])

        # time.sleep(0.05)
    env.close()
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(t_list, q_des_dict['J1'], 'r')
    plt.plot(t_list, q_dict['J1'], 'k')
    plt.legend(['Desired Trajectory','Measured Trajectory'],loc='upper right')
    plt.subplot(212)
    plt.plot(t_list, q_des_dict['J2'], 'r')
    plt.plot(t_list, q_dict['J2'], 'k')
    plt.legend(['Desired Trajectory','Measured Trajectory'],loc='upper right')

    plt.show()
    if saveFig:
        fig.savefig(path.join(savePath,'trajectory.png'))


    return q_dict, qdot_dict, qddot_dict, a_dict