import numpy as np
import platform
import matplotlib.pyplot as plt
from loadModel import load_model, save_model, get_model
from Net import *
from os import path
from Controller import Dynamic_Controller, PD_Dynamic_Controller, PD_Controller
from Trajectory import CosTraj, runTrajectory
pi = np.pi


def loop_func(netType, root_path):
    save_path = path.join(root_path, netType)
    load_path = save_path
    device = torch.device('cpu')
    model = get_model(netType, device)
    model, input_scaler, output_scaler = load_model(path.join(load_path,'model'), netType, model)

    dynamic_controller = Dynamic_Controller(model, input_scaler, output_scaler)
    pd_controller = PD_Controller()
    pd_dynamic_controller = PD_Dynamic_Controller(pd_controller, dynamic_controller)
    traj = CosTraj()
    traj.A = 1
    _, _, _, _, _ = runTrajectory(pd_dynamic_controller, traj, sampleNum = 2000, savePath=save_path,saveFig=True,
                                                          dt=0.01, isShowPlot=True,isRender=False,saveName='testTrajectory',isReturnAllForce=True)

    print("finish test script!")


root_path = path.join('.', 'data', 'trackTrajectory')
netType = 'DeLan'
loop_func(netType, root_path=root_path)




