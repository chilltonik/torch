import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NetGirl(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        return x


# def forward(inp, l1: nn.Linear, l2: nn.Linear):
#     u1 = l1.forward(inp)
#     s1 = F.tanh(u1)
#
#     u2 = l2.forward(s1)
#     s2 = F.tanh(u2)
#     return s2
#
#
# layer1 = nn.Linear(in_features=3, out_features=2)
# layer2 = nn.Linear(2, 1)
#
# print(layer1.weight)
# print(layer1.bias)
#
# layer1.weight.data = torch.tensor([[0.7402, 0.6008, -1.3340], [0.2098, 0.4537, -0.7692]])
# layer1.bias.data = torch.tensor([0.5505, 0.3719])
#
# layer2.weight.data = torch.tensor([[-2.0719, -0.9485]])
# layer2.bias.data = torch.tensor([-0.1461])
#
# x = torch.FloatTensor([1, -1, 1])
# y = forward(x, layer1, layer2)
# print(y.data)

model = NetGirl(3, 2, 1)
model1 = NetGirl(3, 5, 1)
model2 = NetGirl(100, 18, 10)

print(model)
gen_p = model.parameters()
print(list(gen_p))

x_train = torch.tensor([
    (-1, -1, -1),
    (-1, -1, 1),
    (-1, 1, -1),
    (-1, 1, 1),
    (1, -1, -1),
    (1, -1, 1),
    (1, 1, -1),
    (1, 1, 1)
], dtype=torch.float32)

y_train = torch.tensor([-1, 1, -1, 1, -1, 1, -1, -1], dtype=torch.float32)


optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

model.train()

for _ in range(1000):
    k = random.randint(0,  len(y_train)- 1)
    y = model(x_train[k])
    y = y.squeeze()
    loss = loss_func(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

for x, d in zip(x_train, y_train):
    with torch.no_grad(): # or model.requires_grad_(False)
        y = model(x)
        print(f"Output value NN: {y.data} => {d}")
