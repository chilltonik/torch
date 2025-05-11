import torch
from random import randint

def act(z):
    return torch.tanh(z)

def df(z):
    s = act(z)
    return 1 - s * s

def go_forward(x_inp, w1, w2):
    z1 = torch.mv(w1[:, :3], x_inp) + 1*w1[:, 3]
    s1 = act(z1)

    z2 = torch.dot(w2[:2], s1) + 1*w2[2]
    y = act(z2)
    return y, z1, s1, z2


torch.manual_seed(1)

W1 = torch.rand(8).view(2, 4) - 0.5
W2 = torch.rand(3) -0.5

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

lmd = 0.05
N = 1000
total = len(y_train)


for _ in range(N):
    k = randint(0, total - 1)
    x = x_train[k]

    y, z1, s1, out = go_forward(x, W1, W2)

    e = y - y_train[k] # производная квадратичной функции потерь
    delta = e*df(out) # вычисление локального градиента
    delta2 = W2[:2] * delta * df(z1) # вуктор из двух локальных градиентов скрытого слоя

    W2[:2] = W2[:2] - lmd * delta * s1 # корректирровка весов связи последнего слоя
    W2[2] = W2[2] - lmd * delta # корректировка bias

    # корректировка связей первого слоя
    W1[0, :3] = W1[0, :3] - lmd * delta2[0] * x
    W1[1, :3] = W1[1, :3] - lmd * delta2[1] * x

    # корректировка bias
    W1[0, 3] = W1[0, 3] - lmd*delta2[0]
    W1[1, 3] = W1[1, 3] - lmd*delta2[1]


for x, d in zip(x_train, y_train):
    y, z1, s1, out = go_forward(x, W1, W2)
    print(f"Out Nn value: {y} => {d}")

print(f"W1:\n{W1}")
print(f"W2:\n{W2}")
    