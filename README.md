# Thomas Wüthrich - EPFL-Master-Project - District scale prediction of subsurface waste heat flows
Further exploration of ground temperature prediction using machine and deep learning approaches.<br>
This repository contains the files used for the further exploration of the prediction of subsurface temperature distribution used for the EPFL Master thesis in Geotechnics "District scale prediction of subsurface waste heat flows".

## Usage instructions
The available data from finite element simulations is available in the Time_Data.mat file. It's data and a visualization example can be found by running the Visualize_Data.py file. <br>
The main python file can be run in the folder as-is. It uses the PyTorch Residual Network style convolutional neural network defined in the Model.py file. The model is defined as explained below. Note that the model input consists solemnly of the initial temperature distribution and a map describing the location of the heat sources.<br>
The training procedure produces a checkpoint file, which can be used to continue training if wished so. Note that the optimizer is reset when rerunning, even when continuing training! CNN_ResNet_checkpoint.pth provides a pretrained version of a model which can be refined by continuing training or used as-is for validation and evaluation. 

## Model architecture
The Model uses a sequence of convolutional submodels, each submodel constructed as illustrated below (Created with https://alexlenail.me/NN-SVG/LeNet.html) with 2 convolutional layers and a residual network block containing 4 convolutions in between. Each submodel yields the next predicted heat map, which is subsequently recombined with the heat source map and reused to predict the following time step. The submodels for the prediction of time steps <img src="https://render.githubusercontent.com/render/math?math=t > 5"> share the weights pairwise (i.e. 2 subsequent submodels share weights). Kernel Size is 3 for all layers.
<img src="https://github.com/Unundium/Thomas-Wuthrich---EPFL-Master-Project---District-scale-predictionof-subsurface-waste-heat-flows/blob/master/Submodel_architecture.png" width="70%">

## Correspondence for further information to:<br>
thomas.wuthrich@alumni.epfl.ch<br>
joel.zbinden@alumni.epfl.ch
