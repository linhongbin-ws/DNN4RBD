import time
import matplotlib.pyplot as plt
from os import remove
import numpy as np
from os import path
from loadModel import load_model

def train(model, train_loader, valid_loader, loss_fn, optimizer, early_stopping, save_path, max_training_epoch, goal_loss, is_plot=True):
    avg_train_losses = []  # to track the average training loss per epoch as the data trains
    avg_valid_losses = []  # to track the average validation loss per epoch as the data trains
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
        print('Epoch', t, ': Train Loss is ', train_loss, 'Validate Loss is', valid_loss, ", One Interation take ",
              int(ellapse_time), 'seconds')

        if valid_loss <= goal_loss:
            print("Reach goal loss, valid_loss=", valid_loss, '< goal loss=', goal_loss)
            break
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping at Epoch")
            # update the data with checkpoint
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
        fig.savefig(path.join(save_path, 'trainLoss.png'))
