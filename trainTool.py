import torch
import torch.nn.functional as F
import numpy as np
from os import remove
import platform
if platform.system()=='Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from regularizeTool import EarlyStopping
# from Net import Lagrange_Net
from loadModel import load_model

class AutoEncoder(torch.nn.Module):
    def __init__(self, linear_param_generator, act_func, device):
        super(AutoEncoder, self).__init__()
        self.act_func = act_func
        self.w = next(linear_param_generator)
        self.w.requires_grad = True
        self.b = next(linear_param_generator)
        self.b.requires_grad = True
        self.b_T = torch.randn(1, self.w.shape[1], device=device, requires_grad=True)

    def forward(self, x):
        h = F.linear(x, self.w, self.b)
        h = self.act_func(h)
        h = F.linear(h, self.w.t(), self.b_T)
        return h

def train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch, goal_loss, is_plot=True):
    train_losses = []
    valid_losses = []  # to track the validation loss as the model trains
    avg_train_losses = []  # to track the average training loss per epoch as the model trains
    avg_valid_losses = []  # to track the average validation loss per epoch as the model trains

    for t in range(max_training_epoch):
        train_losses = []
        valid_losses = []
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
        print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss)

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
        # fig.savefig(pjoin('model','LogNet',model_file_name+'.png'), bbox_inches='tight')

    return model

def multiTask_train(modelList, train_loaderList, valid_loaderList, optimizer, loss_fn, early_stopping, max_training_epoch, is_plot=True):
    train_losses = []
    valid_losses = []  # to track the validation loss as the model trains
    avg_train_losses = []  # to track the average training loss per epoch as the model trains
    avg_valid_losses = []  # to track the average validation loss per epoch as the model trains
    task_num = len(modelList)
    for t in range(max_training_epoch):
        train_losses = []
        valid_losses = []
        loss = []
        for i in range(task_num):
            for feature, target in train_loaderList[i]:
                model = modelList[i]
                target_hat = model(feature)
                loss = loss_fn(target_hat, target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                train_losses.append(loss.item())

        for i in range(task_num):
            for feature, target in valid_loaderList[i]:
                model = modelList[i]
                target_hat = model(feature)
                loss = loss_fn(target_hat, target)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss)

        early_stopping(valid_loss, modelList)
        if early_stopping.early_stop:
            print("Early stopping at Epoch")
            break
    modelList, _, _ = load_model('.', 'checkpoint', modelList)
    checkpoint = torch.load('checkpoint.pt')
    remove('checkpoint.pt')

    # plot
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
        # fig.savefig(pjoin('model','LogNet',model_file_name+'.png'), bbox_inches='tight')

    return modelList



def pretrain(model, train_loader, valid_loader, learning_rate, earlyStop_patience, max_training_epoch):
    train_losses = []
    valid_losses = []  # to track the validation loss as the model trains
    avg_train_losses = []  # to track the average training loss per epoch as the model trains
    avg_valid_losses = []  # to track the average validation loss per epoch as the model trains

    # number of sequence list = 2*layer_num+1
    squence_list = list(model.children())
    layer_num = int((len(squence_list) + 1) / 2)

    based_layer_list = []
    based_model = []

    # pretrain n-1 layer
    for k in range(layer_num - 1):
        if len(based_layer_list)!=0:
            based_model = torch.nn.Sequential(*based_layer_list)
            for param in based_model.parameters():
                param.requires_grad = False

        train_layer_list = squence_list[2 * k:2 * k + 2]
        # original_generator = train_layer_list[0].parameters()
        # origin_w = next(original_generator)
        # origin_b = next(original_generator)
        train_model = AutoEncoder(train_layer_list[0].parameters(), train_layer_list[1])

        train_params = [train_model.w, train_model.b, train_model.b_T]
        optimizer = torch.optim.Adam(train_params, lr=learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.SmoothL1Loss()
        early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

        for t in range(max_training_epoch):
            train_losses = []
            valid_losses = []
            for feature, _ in train_loader:
                if isinstance(based_model, list):
                    y_hat = feature
                else:
                    y_hat = based_model(feature)
                z_hat = train_model(y_hat)
                loss = loss_fn(y_hat, z_hat)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                train_losses.append(loss.item())

            for feature, target in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                if isinstance(based_model, list):
                    y_hat = feature
                else:
                    y_hat = based_model(feature)
                z_hat = train_model(y_hat)
                loss = loss_fn(y_hat, z_hat)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss)

            early_stopping(valid_loss, train_model)
            if early_stopping.early_stop:
                print("Early stopping at Epoch")
                break
        train_model.load_state_dict(torch.load('checkpoint.pt'))
        remove('checkpoint.pt')

        # plot
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
        # fig.savefig(pjoin('model','LogNet',model_file_name+'.png'), bbox_inches='tight')


        based_layer_list.extend(train_layer_list[0:2])

    based_layer_list.append(squence_list[-1])
    model = torch.nn.Sequential(*based_layer_list)
    return model


