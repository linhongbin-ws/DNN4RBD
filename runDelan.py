from gym_acrobot.envs.acrobot import AcrobotBmt_Dynamics
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import platform
if platform.system()=='Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from os import remove
from reference.regularizeTool import EarlyStopping
from loadModel import load_model, save_model
from DeLan import DeLanNet_inverse
from Net import *
import time



pi = np.pi


valid_ratio = 0.2
batch_size = 512
max_training_epoch = 1000
is_plot = True
goal_loss = 1e-4
earlyStop_patience = 8
learning_rate = 0.04
weight_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Ld_Net = ReLuNet(2,[10,10],2).to(device)
lo_size = 0
for i in range(1, 2):
    lo_size += i
Lo_Net = ReLuNet(2,[10,10],lo_size).to(device)
gNet = ReLuNet(2,[10],2).to(device)
class CustomDataset(Dataset):
    def __init__(self, input_mat, output_mat, is_scale=True, device='cpu'):
        # scale output to zeroscore
        if is_scale:
            self.input_scaler = preprocessing.StandardScaler().fit(input_mat)
            self.output_scaler = preprocessing.StandardScaler().fit(output_mat)
            input_mat = self.input_scaler.transform(input_mat)
            output_mat = self.output_scaler.transform(output_mat)

        # numpy to torch tensor
        self.x_data = torch.from_numpy(input_mat).to(device).float()
        self.y_data = torch.from_numpy(output_mat).to(device).float()

        # get length of pair CAD_sim_1e6
        self.len = self.x_data.shape[0]

        # get dimension of input and output CAD_sim_1e6
        self.input_dim = input_mat.shape[1]
        self.output_dim = output_mat.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




dynamicModel = AcrobotBmt_Dynamics()
model = DeLanNet_inverse(Ld_Net, Lo_Net, gNet, 2,  device=device)
q1_sample_num, q2_sample_num, qd1_sample_num, qd2_sample_num, qdd1_sample_num, qdd2_sample_num = 10, 10, 10, 10, 1, 1

q1_arr = np.linspace(-pi, pi, num=q1_sample_num)
q2_arr = np.linspace(-pi, pi, num=q2_sample_num)
qd1_arr = np.linspace(1, 4, num=q2_sample_num)
qd2_arr = np.linspace(1, 4, num=q2_sample_num)
qdd1_arr = np.linspace(2, 10, num=qdd1_sample_num)
qdd2_arr = np.linspace(2, 10, num=qdd2_sample_num)

total_size = q1_arr.size*q2_arr.size*qd1_arr.size*qd2_arr.size*qdd1_arr.size*qdd2_arr.size
input_mat = np.zeros((total_size, 6))
output_mat = np.zeros((total_size, 2))
cnt = 0
for q1 in q1_arr:
    for q2 in q2_arr:
        for qd1 in qd1_arr:
            for qd2 in qd2_arr:
                for qdd1 in qdd1_arr:
                    for qdd2 in qdd2_arr:
                        s_augmented = [q1, q2, qd1, qd2, qdd1, qdd2, 0.]
                        tau1, tau2 = dynamicModel.inverse(s_augmented)
                        input_mat[cnt,0], input_mat[cnt,1],input_mat[cnt,2], input_mat[cnt,3], input_mat[cnt,4], input_mat[cnt,5] = q1, q2, qd1, qd2, qdd1, qdd2
                        output_mat[cnt, 0], output_mat[cnt, 1] = tau1, tau2
                        cnt +=1
print(input_mat.shape)
print(output_mat.shape)


dataset = CustomDataset(input_mat, output_mat, is_scale=True, device=device)


import os
file_path = os.path.join(".","model")
file_name = "DelanNet"
save_model(file_path, file_name, model, input_scaler=dataset.input_scaler, output_scaler=dataset.output_scaler)


train_ratio = 1 - valid_ratio
train_size = int(dataset.__len__() * train_ratio)
test_size = dataset.__len__() - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          num_workers=0,
                          shuffle=True
                          )
valid_loader = DataLoader(validate_dataset,
                          batch_size=batch_size,
                          num_workers=0,
                          shuffle=True)
loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

avg_train_losses = []  # to track the average training loss per epoch as the model trains
avg_valid_losses = []  # to track the average validation loss per epoch as the model trains

print("test forward ellapse time")
for feature, target in train_loader:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    target_hat = model(feature[0,:].unsqueeze(0))
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print(start.elapsed_time(end))


print("Start Training")
for t in range(max_training_epoch):
    train_losses = []
    valid_losses = []
    start_time = time.time()
    for feature, target in train_loader:
        target_hat = model(feature)
        loss = loss_fn(target_hat, target)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        train_losses.append(loss.item())
    for feature, target in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        target_hat = model(feature)
        loss = loss_fn(target_hat, target)
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    ellapse_time = time.time() - start_time
    print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss,", One Interation take ", ellapse_time)

    if valid_loss<=goal_loss:
        print("Reach goal loss, valid_loss=", valid_loss,'< goal loss=', goal_loss)
        break
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping at Epoch")
        # update the model with checkpoint
        break

model, _, _ = load_model('.', 'checkpoint', model)
remove('checkpoint.pt')
save_model(file_path, file_name, model, input_scaler=dataset.input_scaler, output_scaler=dataset.output_scaler)

### plot the train loss and validate loss curves
if is_plot:
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(max(avg_valid_losses), max(avg_valid_losses)))  # consistent scale
    plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()