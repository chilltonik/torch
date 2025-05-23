import torch
import matplotlib.pyplot as plt

N = 5
b = 3


x1 = torch.rand(N)
x2 = x1 + torch.randint(1, 10, [N]) / 10 + b
C1 = torch.vstack([x1, x2, torch.ones(N)]).mT
print("x1:\n", x1)
print("x2:\n", x2)
print("C1:\n", C1)

x1 = torch.rand(N)
x2 = x1 - torch.randint(1, 10, [N]) / 10 + b
C2 = torch.vstack([x1, x2, torch.ones(N)]).mT
print("x1:\n", x1)
print("x2:\n", x2)
print("C2:\n", C1)

f = [0 + b, 1 + b]
w1 = -0.5
w2 = -w1
w3 = -b * w2 # or 0

w = torch.FloatTensor([w1, w2, w3])
for i in range(N):
    x = C2[:][i]
    y = torch.dot(w, x)
    if y >= 0:
        print("Class C1")
    else:
        print("Class C2")    

plt.scatter(C1[:, 0], C1[:, 1], s=10, c='red')
plt.scatter(C2[:, 0], C2[:, 1], s=10, c='blue')
plt.plot(f)
plt.grid()
plt.show()


