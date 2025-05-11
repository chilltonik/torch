import torch
from torchvision.datasets import ImageFolder

from mnistN_classification import DigitNN, transforms, data, optimizer


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DigitNN(28 * 28, 32, 10)


t = torch.tensor([1, 2, 3], dtype=torch.int8, device=device)
torch.save(t, 'mnist/models/tensor.tar')

tt = torch.load('mnist/models/tensor.tar', weights_only=True, map_location='cpu')
print(t)
print(tt)


st = torch.load('mnist/models/model_1.tar', weights_only=True)

transforms.load_state_dict(st['tfs'])
optimizer.load_state_dict(st['opt'])
model.load_state_dict(st['model'])

print(model)
print(optimizer)
print(transforms)


d_test = ImageFolder("mnist/dataset/test", transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
Q = 0

model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)


