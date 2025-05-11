import torch
import torch.nn as nn


class ModelNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=3, act_type=None):
        super().__init__()
        self.layers_list = [nn.Linear(input_dim // n, input_dim // (n + 1)) for n in range(1, n_layers+1)]
        self.act_type = act_type

        self.layers = nn.ModuleList(self.layers_list) # or
        self.net = nn.Sequential(*self.layers)

        self.layers2 = nn.ModuleList()
        for n in range(1, n_layers + 1):
            self.layers2.add_module(f'layer_{n}', nn.Linear(input_dim // n, input_dim // (n + 1)))

        # sz_input = self.layers[-1].out_features
        self.layer_out = nn.Linear(input_dim // (n_layers + 1), output_dim)

        self.act_lst = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lk_relu': nn.LeakyReLU(),
        })


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.act_type and self.act_type in self.act_lst:
                x = self.act_lst[self.act_type](x)

        x = self.layer_out(x)
        return x



model = nn.Sequential(
    nn.Linear(28*28, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)
# or
model = nn.Sequential()
model.add_module('layer_1',  nn.Linear(28*28, 32))
model.add_module('relu',   nn.ReLU())
model.add_module('layer_2',   nn.Linear(32, 10))

print(model)


block = nn.Sequential(
    nn.Linear(32, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.LeakyReLU()
)

model = ModelNN(28*28, 10, act_type='relu')
print(len(list(model.parameters())))
print(model)



