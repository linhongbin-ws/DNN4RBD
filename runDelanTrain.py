import numpy as np
import matplotlib.pyplot as plt
from os import remove
from reference.regularizeTool import EarlyStopping
from loadModel import load_model, save_model, get_model
from Net import *
import time
from loadData import createLoader
from Trajectory import runTrajectory
from os import path
from Controller import PD_Controller
from Trajectory import CosTraj


pi = np.pi

save_path = path.join('.','data','cos')
netType = 'DeLan'
save_name = netType
valid_ratio = 0.2
batch_size = 512
max_training_epoch = 1000
is_plot = True
goal_loss = 1e-4
earlyStop_patience = 8
learning_rate = 0.04
weight_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
controller = PD_Controller()
traj = CosTraj()
traj.A = 1


model = get_model(netType, device)
q_dict, qdot_dict, qddot_dict, a_dict = runTrajectory(sampleNum = 200, controller = controller, traj=traj, savePath=save_path)
input_mat = np.array([q_dict['J1'],q_dict['J2'],qdot_dict['J1'],qdot_dict['J2'],qddot_dict['J1'],qddot_dict['J2']]).transpose()
output_mat = np.array([a_dict['J1'],a_dict['J2']]).transpose()
print(input_mat.shape)
print(output_mat.shape)
np.savez(path.join(save_path,'trainData'), input_mat, output_mat)


# dataset = CustomDataset(input_mat, output_mat, is_scale=True, device=device)


#
# file_path = os.path.join(".","data")
# file_name = "DelanNet"
# save_model(file_path, file_name, data, input_scaler=dataset.input_scaler, output_scaler=dataset.output_scaler)


train_loader, valid_loader, input_scaler, output_scaler = createLoader(input_mat, output_mat,batch_size,valid_ratio,is_scale=True,device=device)

loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

avg_train_losses = []  # to track the average training loss per epoch as the data trains
avg_valid_losses = []  # to track the average validation loss per epoch as the data trains

# print("test forward ellapse time")
# for feature, target in train_loader:
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#
#     start.record()
#     target_hat = data(feature[0,:].unsqueeze(0))
#     end.record()
#     # Waits for everything to finish running
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))


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
        # forward pass: compute predicted outputs by passing inputs to the data
        target_hat = model(feature)
        loss = loss_fn(target_hat, target)
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    ellapse_time = time.time() - start_time
    print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss,", One Interation take ", int(ellapse_time),'seconds')

    if valid_loss<=goal_loss:
        print("Reach goal loss, valid_loss=", valid_loss,'< goal loss=', goal_loss)
        break
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping at Epoch")
        # update the data with checkpoint
        break

model, _, _ = load_model('.', 'checkpoint', model)
remove('checkpoint.pt')
save_model(path.join(save_path, 'model'), 'DeLan', model, input_scaler=input_scaler, output_scaler=output_scaler)

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
    fig.savefig(path.join(save_path, 'trainLoss.png'))