import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs

from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()


class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(
                map(lambda _x: (os.path.join(path, _x), _target), list_files)
            )

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


model = DigitNN(28 * 28, 32, 10)

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.Grayscale(),
        tfs.ToDtype(torch.float32, scale=True),
        RavelTransform(), # or tfs.Lambda(lambda _img: _img.ravel()),
    ]
)
# d_train = DigitDataset("mnist/dataset", transform=to_tensor) or
# d_train = ImageFolder("mnist/dataset/train", transform=transforms)
# train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

dataset_mnist = torchvision.datasets.MNIST(r"mnist/dataset", download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)



optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

epochs = 20 # 2

model_state_dict = {
    'tfs': transforms.state_dict(),
    'opt': optimizer.state_dict(),
    'model': model.state_dict()
}

best_loss = 1e10
loss_lst_val = []
loss_lst = []

for _e in range(epochs):
    model.train()

    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)

    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

        train_tqdm.set_description(
            f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}"
        )

    if best_loss > loss_mean * 1.1:
        best_loss = loss_mean
        model_state_dict['model'] = model.state_dict()
        torch.save(model_state_dict, f'mnist/models/model_{_e}.tar')

    model.eval()
    Q_val = 0
    count_val = 0
    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_function(p, y_val)
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")


d_test = ImageFolder("mnist/dataset/test", transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        # y = torch.argmax(y_test, dim=1) or
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)

plt.plot(loss_lst)
plt.plot(loss_lst_val)
plt.grid()
plt.show()
