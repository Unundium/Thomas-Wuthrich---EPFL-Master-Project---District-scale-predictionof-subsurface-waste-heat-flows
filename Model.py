import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from sys import exit

class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
            kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
            kernel_size = kernel_size, padding = (kernel_size - 1) // 2)

    def forward(self, x):
        y = self.conv2(F.relu(self.conv1(x)))
        return F.relu(y + x)


class ResNet(nn.Module):                  # Convolutional Resnet
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.resBlock1 = nn.Sequential(
            *(ResNetBlock(8, 3) for _ in range(4)))
        # T5d
        self.conv3 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.resBlock2 = nn.Sequential(
            *(ResNetBlock(8, 3) for _ in range(4)))
        # T_60d, T_180d
        self.conv5 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.resBlock3 = nn.Sequential(
            *(ResNetBlock(8, 3) for _ in range(4)))
        # T_365d, T_730d
        self.conv7 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.resBlock4 = nn.Sequential(
            *(ResNetBlock(8, 3) for _ in range(4)))
        # T_3650d, T_18250d

    def forward(self,x):
        # T_d format: samples, times, x, y
        T_d = torch.zeros((x.shape[0], 7, x.shape[-2], x.shape[-1]))
        Source_Mask = x[:, -1, :, :]

        x = self.resBlock1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        T_d[:, 0, :, :] = x[:, 0, :, :]     # T_5d

        x = torch.cat((x, Source_Mask.unsqueeze(1)), 1)  # readd S_active

        for T in range(1, 3): # reuse the same blocks for other times
            x = F.relu(self.conv4(self.resBlock2(F.relu(self.conv3(x)))))
            T_d[:, T, :, :] = x[:, 0, :, :]
            x = torch.cat((x, Source_Mask.unsqueeze(1)), 1)

        for T in range(3, 5):
            x = F.relu(self.conv6(self.resBlock3(F.relu(self.conv5(x)))))
            T_d[:, T, :, :] = x[:, 0, :, :]
            x = torch.cat((x, Source_Mask.unsqueeze(1)), 1)

        for T in range(5, 7):
            x = F.relu(self.conv8(self.resBlock4(F.relu(self.conv7(x)))))
            T_d[:, T, :, :] = x[:, 0, :, :]
            x = torch.cat((x, Source_Mask.unsqueeze(1)), 1)

        return T_d


def train_model(model, train_input, train_result, epochs=50,
        continue_training = True, mini_batch_size=5, learning_rate=1e-3):
    """
    Train a model on the train_input and train_result data
    """

    # move all to the GPU if available
    if torch.cuda.is_available():
       model = model.cuda()
       train_input,train_result = train_input.cuda(), train_result.cuda()

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    nb_epochs_finished = 0  # epoch counter for checkpoint

    # if we want to continue training, try loading a checkpoint
    if continue_training:
        try:  # Try to load a checkpoint file
            device = 'cpu'
            if torch.cuda.is_available(): device='cuda'
            checkpoint = torch.load('CNN_ResNet_checkpoint.pth', map_location=torch.device(device))
            nb_epochs_finished = checkpoint['nb_epochs_finished']
            model.load_state_dict(checkpoint['model_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v): state[k] = v.cuda()
            print('Checkpoint loaded with {} epochs finished'.format(
                nb_epochs_finished))
            print('Last MSE Error: {}°C, \nLast Loss: {: 1.2e}\n\n'
                  .format(['{: 1.2e}'.format(np.sqrt(e))
                             for e in checkpoint['last_training_rate']],
                            checkpoint['last_loss']))
            model.train()

        except FileNotFoundError:  # File not found
            print('File not found, starting over from scratch')

        except:  # File found but error while loading
            print('Checkpoint file found but could not be loaded')
            exit(1)

    # Initialize Losses and MSE containers
    losses = torch.ones((train_input.shape[0], 7, epochs))
    MSE = torch.ones((7, epochs))

    print('Training epoch:')
    for e in range(epochs):
        print('{:03d}'.format(e+1), end='...', flush=True)  # Status print
        if (not (e + 1) % 10) and e > 0: print('')

        for b in range(0, train_input.size(0), mini_batch_size):
            input = train_input.narrow(0, b, mini_batch_size)
            results = train_result.narrow(0, b, mini_batch_size)

            T_d = model(input)  # forward pass
            if torch.cuda.is_available(): T_d = T_d.cuda() # T_d to GPU

            # Calculate Loss for each calculated time step
            for idx_loss in range(losses.shape[1]):
                losses[:, idx_loss, e] = criterion(T_d[:, idx_loss, :, :],
                                             results[:, :, :, idx_loss])
            # The MSE is the loss for each time step, real loss needs to
            # sum over models and times
            MSE[:, e] = losses[:, :, e].mean(0)
            loss = losses[:, :, e].sum((0, 1))

            # Zero grad and backward pass
            model.zero_grad()
            loss.backward(retain_graph=True), optimizer.step()

    # collect metrics to return
    metrics = {'Losses': losses.sum((0, 1)).detach().numpy(),
               'MSE': MSE.detach().numpy()}

    # Save the new checkpoint
    checkpoint = {
        'nb_epochs_finished': nb_epochs_finished + epochs,
        'model_state': model.cpu().state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'last_loss': metrics['Losses'][-1],
        'last_training_rate': metrics['MSE'][:, -1]}
    torch.save(checkpoint, 'CNN_ResNet_checkpoint.pth')
    print('\n\nCheckpoint saved with {} epochs'
          .format(nb_epochs_finished + epochs))

    return model, metrics