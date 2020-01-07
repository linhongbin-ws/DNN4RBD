import numpy as np
import platform
import matplotlib.pyplot as plt
from loadModel import load_model, save_model, get_model
from Net import *
from os import path
from Controller import Dynamic_Controller, PD_Dynamic_Controller, PD_Controller
from Trajectory import CosTraj, runTrajectory, ValinaCosTraj
pi = np.pi


def loop_func(netType, root_path):
    # A_dict = {}
    # w_dict = {}
    # b_dict = {}
    # A_dict['J1'] = []
    # A_dict['J2'] = []
    # w_dict['J1'] = []
    # w_dict['J2'] = []
    # b_dict['J1'] = []
    # b_dict['J2'] = []
    # A_dict['J1'].append(0.2); w_dict['J1'].append(20); b_dict['J1'].append(0.2)
    # A_dict['J1'].append(0.3);w_dict['J1'].append(10);b_dict['J1'].append(0.1)
    # A_dict['J1'].append(0.5); w_dict['J1'].append(6); b_dict['J1'].append(0.3)
    # A_dict['J1'].append(0.6);w_dict['J1'].append(2);b_dict['J1'].append(0.4)
    # A_dict['J1'].append(1); w_dict['J1'].append(1); b_dict['J1'].append(0.6)
    #
    # A_dict['J2'].append(0.2); w_dict['J2'].append(20); b_dict['J2'].append(0.3)
    # A_dict['J2'].append(0.3);w_dict['J2'].append(10);b_dict['J2'].append(0.2)
    # A_dict['J2'].append(0.5); w_dict['J2'].append(5); b_dict['J2'].append(0.1)
    # A_dict['J2'].append(0.6);w_dict['J2'].append(2);b_dict['J2'].append(0.3)
    # A_dict['J2'].append(1); w_dict['J2'].append(1); b_dict['J2'].append(0.1)


    save_path = path.join(root_path, netType)
    load_path = save_path
    a = np.load(path.join(load_path,'trainTrajectory.npz'), allow_pickle=True)
    A_dict = a['arr_0'].tolist()
    w_dict = a['arr_1'].tolist()
    b_dict = a['arr_2'].tolist()
    device = torch.device('cpu')
    model = get_model(netType, device)
    model, input_scaler, output_scaler = load_model(path.join(load_path,'model'), netType, model)

    dynamic_controller = Dynamic_Controller(model, input_scaler, output_scaler)
    pd_controller = PD_Controller()
    pd_dynamic_controller = PD_Dynamic_Controller(pd_controller, dynamic_controller)
    traj = ValinaCosTraj(A_list_list=[A_dict['J1'],A_dict['J2']], w_list_list=[w_dict['J1'],w_dict['J2']],
                         b_list_list=[b_dict['J1'],b_dict['J2']])
    q_dict, qdot_dict, qddot_dict, a_dict, _ = runTrajectory(pd_dynamic_controller, traj, sampleNum = 2000, savePath=save_path,saveFig=True,
                                                          sim_hz=100, isShowPlot=True,isRender=False,saveName='testTrajectory',isReturnAllForce=True, isPlotPredictVel=True)


    print("finish test script!")


root_path = path.join('.', 'data', 'trackTrajectory')
# netType = 'DeLan'
# netType = 'DeLan_Sin'
netType = 'DeLanJacobianNet_inverse'
loop_func(netType, root_path=root_path)




