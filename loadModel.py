import numpy as np
from os import path, mkdir
from Net import *
from DeLan import *
def get_model(use_net, device='cpu'):
    ### define net for acrobot
    if use_net == 'DeLan':
        Ld_Net = ReLuNet(2, [10, 10], 2).to(device)
        lo_size = 0
        for i in range(1, 2):
            lo_size += i
        Lo_Net = ReLuNet(2, [10, 10], lo_size).to(device)
        gNet = ReLuNet(2, [10], 2).to(device)
        model = DeLanNet_inverse(Ld_Net, Lo_Net, gNet, 2, device=device)
        return model
    elif use_net == 'DeLan_Sin':
        model = Delan_Sin(2, device)
        return model
    elif use_net == 'DeLanJacobianNet_inverse':
        model = DeLanJacobianNet_inverse(2, device)
        return model
    else:
        raise Exception(use_net + 'is not support')




def save_model(file_path, file_name, model, input_scaler=None, output_scaler=None):
    if not path.exists(file_path):
        mkdir(file_path)

    if isinstance(model, list):
        save_dict = {'data' + str(i + 1): model[i].state_dict() for i in range(len(model))}
    else:
        save_dict = {'data': model.state_dict()}

    if input_scaler is not None:
        save_dict['input_scaler'] = input_scaler
    if output_scaler is not None:
        save_dict['output_scaler'] = output_scaler

    torch.save(save_dict, path.join(file_path, file_name+'.pt'))

def load_model(file_path, file_name, model):
    file = path.join(file_path, file_name)
    file = file+'.pt'
    if not path.isfile(file):
        raise Exception(file+ 'cannot not be found')

    checkpoint = torch.load(file)
    if isinstance(model, list):
        for i in range(len(model)):
            model[i].load_state_dict(checkpoint['data' + str(i + 1)])
    else:
        model.load_state_dict(checkpoint['data'])

    if 'input_scaler' in checkpoint:
        input_scaler = checkpoint['input_scaler']
    else:
        input_scaler = None

    if 'output_scaler' in checkpoint:
        output_scaler = checkpoint['output_scaler']
    else:
        output_scaler = None
    return model, input_scaler, output_scaler