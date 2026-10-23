import torch
from torch import nn

from local_code.base_class.method import method


class Method_MLP_Stage2(method, nn.Module):
    def __init__(self, hidden_size=128, second_hidden_size=0):
        method.__init__(self, 'Stage 2 MLP', 'MNIST digit classifier')
        nn.Module.__init__(self)

        self.fc_layer_1 = nn.Linear(784, hidden_size)
        self.activation_func = nn.ReLU()
        self.second_hidden_size = second_hidden_size

        if second_hidden_size > 0:
            self.fc_layer_2 = nn.Linear(hidden_size, second_hidden_size)
            self.output_layer = nn.Linear(second_hidden_size, 10)
        else:
            self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.activation_func(self.fc_layer_1(x))
        if self.second_hidden_size > 0:
            x = self.activation_func(self.fc_layer_2(x))
        return self.output_layer(x)
