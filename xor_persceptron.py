import torch

def act(x):
    return 0 if x <= 0 else 1

w_hidden = torch.tensor([[1, 1, -1.5], [1, 1, -0.5]], dtype=torch.float16)
w_out = torch.tensor([-1, 1, -0.5], dtype=torch.float16)

# C1 = [(1, 0), (0, 1)]
# C2 = [(0, 0), (1, 1)]

data_x = [1, 0]
x = torch.tensor(data_x + [1], dtype=torch.float16)
z_hidden = torch.matmul(w_hidden, x)
print("Z hidden:\n", z_hidden)

u_hidden = torch.tensor([act(x) for x in z_hidden] + [1], dtype=torch.float16)
print("U hidden:\n", u_hidden)

z_out = torch.dot(w_out, u_hidden)
y = act(z_out)
print("Y:\n", y)
