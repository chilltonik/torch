import torch
import numpy as np

# basic
# t = torch.Tensor(3, 5, 2)
# # print(t)

# t = torch.empty(3, 5, 2) # better
# print(t)
# print(t.dtype)

# # t = torch.tensor([1]) # if data is known
# print(t.type())

# print(t.dim())
# print(t.size())
# print(t.shape)

# d_np = np.array([[1, 2, 3], [4, 5, 6]])
# print(d_np)

# t2 = torch.from_numpy(d_np)
# print(t2)

# t3 = torch.tensor(d_np, dtype=torch.float32) # copy d_np
# print(t3)

# d = t2.numpy()
# print(d)


# autofill, reshape

tz = torch.zeros(2, 3, dtype=torch.int32)
print(tz)

t = torch.ones(2, 3, dtype=torch.long)
print(t)

t = torch.eye(3)
print(t)

t = torch.eye(3, 2)
print(t)

t = torch.full((2, 4), 5)
print(t)


t = torch.arange(7)
print(t)

t = torch.arange(-5, 0)
print(t)

t = torch.arange(-5, 0, 2)
print(t)

t = torch.arange(1, 0, -0.2)
print(t)

t = torch.linspace(1, 5, 4)
print(t)


t = torch.rand(2, 3)
print(t)

t = torch.randn(2, 3)
print(t)

torch.manual_seed(12)


x = torch.FloatTensor(2, 4).fill_(1)
print(x)

x.uniform_(0, 1)
print(x)

x.normal_(0, 1)
print(x)


x = torch.arange(27)
d = x.view(3, 9)
print(d)

x[0] = 100
print(x)
print(d)

r = x.reshape(3, 3, 3)
print(r)

x.resize_(2, 3)
print(x)

x.ravel()
print("DD", x)

print(d)
d.permute(1, 0)
print(d)

print(d.mT)


x_test = torch.arange(32).view(8, 2, 2)
print(x_test)

x_test4 = torch.unsqueeze(x_test, dim=0)
print(x_test4)

r = x_test.unsqueeze(0)
print(r.size())

x_test.unsqueeze_(0)
print(x_test.size())

b = torch.unsqueeze(x_test4, dim=-1)
print(b.size())

c = torch.squeeze(b)
print(c.size())


# Indexing and slices
a = torch.arange(12)
print(a[2])
print(a[2].item())

print(a[-2])
a[0] = 100
print(a)

b = a[2:4] # new view
print(b) 

a[:4] = torch.IntTensor([-1, -2, -3, -4])
print(a)

x = torch.IntTensor([
    [1, 2, 3],
    [10, 20, 30],
    [100, 200, 300]
])
print(x[1, 1])
print(x[-1, -1])

print(x[0])
print(x[0, :])
print(x[:, 1])

print(a[[0]]) # copy

print(a[a>5])

# MATH
a = torch.FloatTensor([1, 2, 3])
print(a)
print(a-3)

b = torch.IntTensor([3, 4, 5])
print(a+b) 

# Trigonometry and statistics
a = torch.FloatTensor([1, 2, 3, 10, 20, 30])
print(a)
print(a.sum())
print(a.mean())
print(a.max())

a = a.view(3, 2)
print(a)
print(a.sum(dim=0))
print(a.amax(dim=1))

# matrixes
a = torch.arange(1, 10).view(3, 3)
b = torch.arange(10, 19).view(3, 3)

r1 = a*b
print(r1)

c = torch.matmul(a, b) # or mm() without transliation
print(c)

v = torch.LongTensor([-1, -2, -3])
c = torch.matmul(a, v)
print(c)
c = torch.matmul(v, a)
print(c)

# c = torch.mm(v, a) # error
c = a.mm(b)
print(c)
c = a.matmul(b)
print(c)

# butches
bx = torch.randn(7, 3, 5)
by = torch.randn(7, 5, 4)

bc = torch.bmm(bx, by)
print(bc)


a  = torch.arange(1, 10, dtype=torch.float32)
b = torch.ones(9)

c = torch.dot(a, b)
print(c)

c = torch.outer(a, b)
print(c)

a = torch.FloatTensor([1, 2, 3])
b = torch.arange(4, 10, dtype=torch.float32).view(2, 3)

r = torch.mv(b, a)
print(r)

# linalg
a = torch.FloatTensor([(1, 2, 3), (1, 4, 9), (1, 8, 27)])
print(torch.linalg.matrix_rank(a))

y = torch.FloatTensor([10, 20, 30])

print(torch.linalg.solve(a, y))

invA = torch.linalg.inv(a)
x = torch.mv(invA, y)
print(x)




