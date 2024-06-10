# StatQuest with Josh Starmer code

# create tensors to store numerical values
import torch
# make the weight and biases part of nn
import torch.nn as nn
# for the activation functions
import torch.nn.functional as F
# to use adam optimizer
from torch.optim import Adam

# training easy to code
import lightning as L
# we need this to lightning
from torch.utils.data import TensorDataset, DataLoader

class LSTMbyHand(L.LightningModule):
    # create and initialize weight and bias
    def __init__(self):
        super().__init__() 
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
        # create two weights with normal distribution and let requires_grad as True because we want to optimize them
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        # Bias = 0, and requires_grad = True because we want to optimize it
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # others weights and biases of the LSTM
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # does the lstm math  
    def lstm_unit(self, input_value, long_memory, short_memory):
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) + 
                                               self.blr1)
        
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) + 
                                                    self.bpr1)
        
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                       (input_value * self.wp2) +
                                        self.bp1)
        
        updated_long_memory = ((long_memory * long_remember_percent) +
                                (potential_remember_percent * potential_memory))
          
    # make a forward pass through unrolled LSTM
    def forward(self, input):
    
    # configure Adam optimizer
    def configure_optimizers(self):
    
    # calculate the loss and log training progress using training_step()
    def training_step(self, batch, batch_idx):
        