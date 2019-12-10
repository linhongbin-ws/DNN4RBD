import gym
from gym_acrobot.envs.acrobot import AcrobotBmt_Dynamics
import time
import numpy as np
pi = np.pi


model = AcrobotBmt_Dynamics()

q1_sample_num, q2_sample_num, qdd1_sample_num, qdd2_sample_num = 10, 10, 10, 10



q1_arr = np.linspace(-pi, pi, num=q1_sample_num)
q2_arr = np.linspace(-pi, pi, num=q2_sample_num)
qdd1_arr = np.linspace(-10, 10, num=qdd1_sample_num)
qdd2_arr = np.linspace(-10, 10, num=qdd2_sample_num)

total_size = q1_arr.size*q2_arr.size*qdd1_arr.size*qdd2_arr.size
input_mat = np.zeros((total_size, 6))
output_mat = np.zeros((total_size, 2))
cnt = 0
for q1 in range(q1_arr.size):
    for q2 in range(q2_arr.size):
        for qdd1 in range(qdd1_arr.size):
            for qdd2 in range(qdd2_arr.size):
                s_augmented = [q1, q2, 0., 0., qdd1, qdd2, 0.]
                tau1, tau2 = model.inverse(s_augmented)
                input_mat[cnt,0], input_mat[cnt,1], input_mat[cnt,2], input_mat[cnt,3] = q1, q2, qdd1, qdd2
                output_mat[cnt, 0], output_mat[cnt, 1] = tau1, tau2
                cnt +=1
print(input_mat.shape)
print(output_mat.shape)

