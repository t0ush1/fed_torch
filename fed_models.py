import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class BasicModel:
    def __init__(self, input, output, model_type, model):
        self.input = input
        self.output = output
        self.model_type = model_type
        self.model = model.cuda()
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss().cuda()
        print(f"Now we are creating {model_type} with input={input}, output={output}")

    def fit(self, dataset, epochs, batch_size):
        self.model.train()
        train_x = torch.from_numpy(dataset[0]).float()
        train_y = torch.from_numpy(dataset[1]).long()
        dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        for epoch in range(epochs):
            for data, label in train_loader:
                data, label = data.cuda(), label.cuda()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def load_params(self, params):
        keys = self.model.state_dict().keys()
        values = [torch.from_numpy(nparray.copy()).cuda() for nparray in params]
        self.model.load_state_dict(OrderedDict(zip(keys, values)))

    def get_params(self):
        return [tensor.cpu().numpy() for tensor in self.model.state_dict().values()]

    def eval(self, dataset):
        self.model.eval()
        test_x = torch.from_numpy(dataset[0]).float().cuda()
        test_y = torch.from_numpy(dataset[1]).long().cuda()
        with torch.no_grad():
            output = self.model(test_x)
            loss = self.criterion(output, test_y)
            correct = output.argmax(dim=1).eq(test_y).sum().item()
            accuracy = correct / len(test_y)
        print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        return loss.item(), accuracy


class LinearModel(BasicModel):
    def __init__(self, input, output):
        super().__init__(input, output, "Linear Model", nn.Sequential(
            nn.Flatten(),
            nn.Linear(input[0] * input[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output)
        ))


class CNNModel(BasicModel):
    def __init__(self, input, output):
        super().__init__(input, output, "CNN Model", nn.Sequential(
            Reshape(1, input[0], input[1]),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear((input[0] // 4 // 4) * (input[1] // 4 // 4) * 64, output)
        ))

DEFAULT_EPOCH = 5
DEFAULT_BATCH = 16

SHAPE_DICT = {"emnist": ((28, 28), 10), "mnist": ((28, 28), 10)}
MODEL_DICT = {"linear_model": LinearModel, "cnn_model": CNNModel}
