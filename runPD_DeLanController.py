import numpy as np
import platform
import matplotlib.pyplot as plt
from loadModel import load_model, save_model, get_model
from Net import *
from os import path
from Controller import Dynamic_Controller, PD_Dynamic_Controller, PD_Controller
from Trajectory import CosTraj, runTrajectory


pi = np.pi

netType = 'DeLan'
load_path = path.join('.','data','cos')
save_path = path.join('.','data','cos2')
device = torch.device('cpu')


model = get_model(netType, device)
model, input_scaler, output_scaler = load_model(path.join(load_path,'model'), netType, model)

dynamic_controller = Dynamic_Controller(model, input_scaler, output_scaler)
pd_controller = PD_Controller()
pd_dynamic_controller = PD_Dynamic_Controller(pd_controller, dynamic_controller)

traj = CosTraj()
traj.A = 1
q_dict, qdot_dict, qddot_dict, a_dict = runTrajectory(pd_dynamic_controller, traj, sampleNum = 20000, savePath=save_path,saveFig=True, dt=0.01, isShowPlot=True,isRender=True)







