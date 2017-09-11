import torch
import torch.nn.functional as function
import torch.optim.optimizer as optim
import torch.nn as nn
class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size= 11, stride=4, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, padding=2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(function.relu(self.conv1(x), inplace=True))
        x = self.pool2(function.relu(self.conv2(x), inplace=True))
        x = function.relu(self.conv3(x), inplace=True)
        x = function.relu(self.conv4(x), inplace=True)
        x = self.pool3(function.relu(self.conv5(x), inplace=True))
        x = x.view(x.size(0), 256*6*6)
        x = function.relu(self.fc1(self.dropout1(x)), inplace=True)
        x = function.relu(self.fc2(self.dropout2(x)), inplace=True)
        x = self.fc3(x)
        return x

