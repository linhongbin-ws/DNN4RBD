import numpy as np
from reference.regularizeTool import EarlyStopping
from loadModel import load_model, save_model, get_model
from Net import *
import time
from loadData import createLoader
from Trajectory import runTrajectory
from os import path, mkdir
from Controller import PD_Controller
from Trajectory import CosTraj, ValinaCosTraj
pi = np.pi
from trainTool import train

def loop_func(netType, root_save_path):

    ## hyper parameters for training configuration
    save_path = path.join(root_save_path,netType)
    valid_ratio = 0.2
    batch_size = 512
    max_training_epoch = 1000
    is_plot = True
    goal_loss = 1e-4
    earlyStop_patience = 8
    learning_rate = 0.04
    weight_decay = 1e-4
    sample_ratio = 1
    A_list_list = [[1, 0.2, 0.1, 0.2],    [1, 0.2, 0.1, 0.2]]
    w_list_list = [[1, 3,   5,   6],      [1,   3,   5,   6]]
    b_list_list = [[0, 0,   0,   0],      [0.1, 0.2, 0.3, 0.6]]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using hardware for training model: ",device)
    controller = PD_Controller()
    for i in range(2):
        controller.kp[i] = controller.kp[i]*10
        controller.kd[i] = controller.kd[i] * 2
    traj = ValinaCosTraj(A_list_list=A_list_list, w_list_list=w_list_list,
                         b_list_list=b_list_list)
    sampleNum = 2000

    ## run simulation for acrobot
    model = get_model(netType, device)
    model = model.to(device)
    q_dict, qdot_dict, qddot_dict, a_dict = runTrajectory(sampleNum = sampleNum, controller = controller, traj=traj, sim_hz=100,
                                                          savePath=save_path, isShowPlot=True, isRender=False, saveName='trainTrajectory', sample_ratio=sample_ratio)
    input_mat = np.array([q_dict['J1'],q_dict['J2'],qdot_dict['J1'],qdot_dict['J2'],qddot_dict['J1'],qddot_dict['J2']]).transpose()
    output_mat = np.array([a_dict['J1'],a_dict['J2']]).transpose()
    print(input_mat.shape)
    print(output_mat.shape)
    if not path.isdir(save_path):
        mkdir(save_path)
    np.savez(path.join(save_path,'trainData'), input_mat, output_mat)
    np.savez(path.join(save_path, 'trainTrajectory'), A_list_list, w_list_list, b_list_list)

    ## data loader for training. good solution for loading big data
    train_loader, valid_loader, input_scaler, output_scaler = createLoader(input_mat, output_mat,batch_size,valid_ratio,is_scale=True,device=device)

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

    train(model, train_loader, valid_loader, loss_fn, optimizer, early_stopping, save_path, max_training_epoch,
          goal_loss, is_plot=True)


    save_model(path.join(save_path, 'model'), netType, model, input_scaler=input_scaler, output_scaler=output_scaler)
    print("Finish training!")

# netType = 'DeLan'
#netType = 'DeLan_Sin'
netType = 'DeLanJacobianNet_inverse'
root_save_path = path.join('.', 'data', 'trackTrajectory')
loop_func(netType, root_save_path)