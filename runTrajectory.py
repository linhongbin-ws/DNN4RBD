import numpy as np
import platform
import matplotlib.pyplot as plt
from loadModel import load_model, save_model, get_model
from Net import *
from os import path
from Controller import Dynamic_Controller, PD_Dynamic_Controller, PD_Controller
from Trajectory import CosTraj, runTrajectory, ValinaCosTraj
pi = np.pi


def loop_func(root_path):
    save_path = path.join(root_path)
    load_path = save_path

    pd_controller = PD_Controller()
    for i in range(2):
        pd_controller.kp[i] = pd_controller.kp[i]*10
        pd_controller.kd[i] = pd_controller.kd[i] * 2
    traj = ValinaCosTraj(A_list_list=[[0.1,0.2,0.4,1], [0.1,0.2,0.4,1]], w_list_list=[[6,4,2,1],[6,4,2,1]], b_list_list=[[0,0,0,0],[0.5,0.5,0.5,0.5]])
    _, _, _, _= runTrajectory(pd_controller, traj, sampleNum = 1600, savePath=save_path,saveFig=True,
                                                          sim_hz=100, isShowPlot=True,isRender=True,saveName='testTrajectory')

    print("finish test script!")


root_path = path.join('.', 'data', 'Trajectory')
loop_func(root_path=root_path)




