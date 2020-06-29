import numpy as np
import os
import scipy.io as io
import torch
from torch import nn
import time
import matplotlib.pyplot as plt

import Model as M

##########################################################################
#######   Functions ######################################################
def load_Data(path=os.path.join(os.getcwd(),'Time_Data')):
    """
    Load Data from mat file and create Pytorch Tensor Data Set

    path: <string> Absolute Path to mat file (without .mat ending).
            Default is current directory with 'Time_Data' file
    """

    data_dict = io.loadmat(path)
    # Format: model, x, y, time
    maps = torch.tensor(data_dict['data'], dtype=torch.float)
    times = torch.tensor(data_dict['times'][0], dtype=torch.float)
    S_active = torch.tensor(data_dict['S_active'], dtype=torch.float)

    # concatenate input data to format model, x, y, (T(t=0), S_active)
    Input_Data = torch.cat((maps[:, :, :, 0].unsqueeze(1),
                            S_active.unsqueeze(1)), 1)

    # Validation data, temperature data at other time steps
    Validation_Data = maps[:, :, :, 1:]  # model, x, y, t

    return Input_Data, Validation_Data, times


##########################################################################
##################  Training parameters  #################################

epochs = 5
training_split = 0.66    # Train/Test split. take care with mini batches
mini_batch_size = 5      # such that the train set can be separated
learning_rate = 1e-6
continue_training = True  # Do you want to start over or continue?


##########################################################################
#########  The secret codes...usually no need to touch  ##################
t0 = time.perf_counter()
# Load Data
Input_Data, Validation_Data, Times = load_Data()

# Prepare data (random split to train & test)
idx = np.random.permutation(range(Input_Data.shape[0]))
limit_index = int(np.ceil(len(idx)*training_split))

Train_Input = Input_Data[idx[:limit_index], :, :, :]
Train_Result = Validation_Data[idx[:limit_index], :, :]

Test_Input = Input_Data[idx[limit_index:], :, :, :]
Test_Result = Validation_Data[idx[limit_index:], :, :]

# Define model and launch training
model = M.ResNet()  # create network instance
model, metrics = M.train_model(model, Train_Input,
        Train_Result, epochs=epochs, continue_training=continue_training,
        mini_batch_size=mini_batch_size, learning_rate=learning_rate)

print('\n\n Total training time: {: .2f}s'.format(time.perf_counter()-t0))

# Validate Model on test set
with torch.no_grad():
    if torch.cuda.is_available():
        model = model.cuda()
        Test_Input, Test_Result = Test_Input.cuda(), Test_Result.cuda()

    crit = nn.MSELoss()
    T_d = model(Test_Input)  # Forward pass on test set

    if torch.cuda.is_available(): T_d = T_d.cuda()

    # Get MSE for models and time periods
    test_MSE = torch.ones((Test_Input.shape[0], 7))
    for idx_model in range(Test_Input.shape[0]):
        for idx_loss in range(test_MSE.shape[1]):
            test_MSE[idx_model, idx_loss] = crit(
                    T_d[idx_model, idx_loss, :, :],
                    Test_Result[idx_model, :, :, idx_loss])

labels = ['T_5d', 'T_60d', 'T_180d', 'T_365d',
            'T_730d', 'T_3650d', 'T_18250d']

fig, axs = plt.subplots(2, 1, tight_layout=True)
fig.canvas.set_window_title('Errors')
axs = axs.reshape(-1)

for idx, (time_MSE, test_MSE_row, label) in enumerate(zip(metrics['MSE'],
                                                test_MSE, labels)):
    axs[0].plot(range(1, epochs+1), np.sqrt(time_MSE), label=label)
    axs[0].set_title('Training RMSE')
    axs[0].legend(), axs[0].grid(b=True)

    axs[1].plot(range(1,8), np.sqrt(test_MSE_row),
                label='Model {}'.format(idx))
    axs[1].set_title('Testing RMSE')
    axs[1].set_xticks(range(1,8)), axs[1].set_xticklabels(labels)
    axs[1].legend(), axs[1].grid(b=True)

# Heat Maps
fig1, axes = plt.subplots(7, 2*Test_Input.shape[0], tight_layout=True)
fig1.canvas.set_window_title('Test maps and targets')
axes = axes.reshape(-1)

for idx in range(Test_Input.shape[0]):
    for idx_time in range(7):
        im = axes[2*Test_Input.shape[0]*idx_time + 2*idx].imshow(
                                        T_d[idx, idx_time, :, :].cpu(),
                                        cmap='jet', origin='lower')
        axes[2*Test_Input.shape[0]*idx_time + 2*idx].set_title(
                                                        labels[idx_time])
        axes[2*Test_Input.shape[0]*idx_time + 2*idx].axis('off')

        axes[2*Test_Input.shape[0]*idx_time + 2*idx + 1].imshow(
                                    Test_Result[idx, :, :, idx_time].cpu(),
                                    cmap='jet', origin='lower')
        axes[2*Test_Input.shape[0]*idx_time + 2*idx + 1].set_title(
                                    'Target {}'.format(labels[idx_time]))
        axes[2*Test_Input.shape[0]*idx_time + 2*idx + 1].axis('off')

# Validate Model on training set
with torch.no_grad():
    if torch.cuda.is_available():
        model = model.cuda()
        Train_Input, Train_Result = Train_Input.cuda(), Train_Result.cuda()
    crit = nn.MSELoss()
    train_MSE = torch.ones((Train_Input.shape[0], 7))
    T_d = model(Train_Input)

    if torch.cuda.is_available(): T_d = T_d.cuda()

    for idx_loss in range(train_MSE.shape[1]):
        train_MSE[:, idx_loss] = crit(T_d[:, idx_loss, :, :],
                                         Train_Result[:, :, :, idx_loss])

fig2, axes = plt.subplots(7, 2*Train_Input.shape[0], tight_layout=True)
fig2.canvas.set_window_title('Train maps and targets')
axes = axes.reshape(-1)

for idx in range(Train_Input.shape[0]):
    for idx_time in range(7):
        img = axes[2*Train_Input.shape[0]*idx_time + 2*idx].imshow(
                                        T_d[idx, idx_time, :, :].cpu(),
                                        cmap='jet', origin='lower')
        axes[2*Train_Input.shape[0] * idx_time + 2*idx].set_title(
                                                        labels[idx_time])
        axes[2*Train_Input.shape[0] * idx_time + 2*idx].axis('off')

        axes[2*Train_Input.shape[0] * idx_time + 2*idx + 1].imshow(
                                Train_Result[idx, :, :, idx_time].cpu(),
                                cmap='jet', origin='lower')
        axes[2*Train_Input.shape[0] * idx_time + 2*idx + 1].set_title(
                                                            'Target')
        axes[2*Train_Input.shape[0] * idx_time + 2*idx + 1].axis('off')

plt.show()