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

# create a LSTM by hand with some methods
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
        # calculates de percentage of the lstm to remember
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) + 
                                               self.blr1)
        
        # creates a new long term memory and determines what percentage to remember
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) + 
                                                    self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                       (input_value * self.wp2) +
                                        self.bp1)
        
        # update the long term memory
        updated_long_memory = ((long_memory * long_remember_percent) +
                                (potential_remember_percent * potential_memory))
        
        # create a new short term memory and determine what percentage to remember
        output_percent = torch.tanh((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                        self.bo1)
        # update the short term memory
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        
        # return the new long and short term memories
        return ([updated_long_memory, updated_short_memory])

    # make a forward pass through unrolled LSTM
    def forward(self, input):
        long_memory = 0
        short_memory = 0

        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        # long_memory = 0, short_memory = 0
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        # long and short memory comes from above
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        # long and short memory comes from above
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        # long and short memory comes from above
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        # retorna o output
        return short_memory
    
    # configure Adam optimizer
    def configure_optimizers(self):
        return Adam(self.parameters())
    # calculate the loss and log training progress using training_step()
    def training_step(self, batch, batch_id):
        input_i, label_i = batch
        # uses the forward method to make a prediction with the training data
        output_i = self.forward(input_i[0])
        # calculates the loss, in this case, the sum of the squared residuals
        loss = (output_i - label_i)**2

        # we are logging the loss so that we can review it later 
        self.log("train_loss", loss)

        # label_i == 0, company A, else, company B
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
    

model = LSTMbyHand()