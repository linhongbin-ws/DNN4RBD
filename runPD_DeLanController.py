import numpy as np
import platform
if platform.system()=='Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from os import remove
from reference.regularizeTool import EarlyStopping
from loadModel import load_model, save_model, get_model
from Net import *
import time
from loadData import createLoader
from Trajectory import genTrajectoryData
from os import path
from Controller import Dynamic_Controller, PD_Dynamic_Controller, PD_Controller


pi = np.pi

netType = 'DeLan'
save_path = path.join('.','data','cos')
device = torch.device('cpu')


model = get_model(netType, device)
model, input_scaler, output_scaler = load_model(path.join(save_path,'model'), netType, model)

dynamic_controller = Dynamic_Controller(model, input_scaler, output_scaler)
pd_controller = PD_Controller()
pd_dynamic_controller = PD_Dynamic_Controller(pd_controller, dynamic_controller)






