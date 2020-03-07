#imports 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np 

#hyperparameters
num_epochs = 100
num_classes = -1
batch_size = -1
learning_rate = 0.001

#set up training & testing data: DataLoader

class PINN(nn.Module):
	def _init_(self):
		super(PINN, self)._init()
		self.cnn = nn.Sequential(
			#Conv2d parameters: # of input channels, # of output channels,
			#size of convolutional filter (square or tuple of x/y), how much 
			#filter moves each iteration, padding (from equation)
			nn.Conv2d(3, -1, kernel_size = -1, stride = 1, padding = -1),
			nn.ReLU(),
			#MaxPool2d parameters: same as Conv2d, except stride is >1 as
			#we want to downsize the image (from equation) 
			nn.MaxPool2d(kernel_size = -1, stride = -1, padding = -1))
		#possible to insert another convolutional layer
		#somehow set up feature vectors: # of nodes in current layer, # 
		#of nodes in next layer
		self.fc1 = nn.Linear(-1, -1)
		#LSTM parameters: input dimension at each time step, size of
		#hidden/cell state at each time step, # of LSTM layers
		self.lstm = nn.LSTM(-1, -1, -1, batch_first = True)
		#avoid overfitting (here, after CNN, or both?)
		self.drop_out = nn.Dropout()
		#go from LSTM output to # of classes: # of nodes in current layer, # 
		#of nodes in next layer 
		self.fc2 = nn.Linear(-1, -1)
		#make class identification between 0 & 1
		self.sigmoid = nn.Sigmoid()


	def forward(self, x, hidden):
		out = self.cnn(x)
		#go from some multidimensional input to one-dimensional output
		out = out.reshape(out.size(0), -1)
		out = self.fc1(out)
		out, hidden = self.lstm(out, hidden)
		#second parameter is size of hidden dimension
		out = out.contiguous().view(-1, -1)
		out = self.dropout(out)
		out = self.fc2(out)
		out = self.sigmoid(out)
		#only take last timestep
		out = out.view(batch_size, -1)
        out = out[:,-1]
		return out, hidden

	def init_hidden(self):
		#Hidden parameters: # of layers, batch size, size of hidden dimension
		return (torch.zeros(-1, -1, -1), torch.zeros(-1, -1, -1))