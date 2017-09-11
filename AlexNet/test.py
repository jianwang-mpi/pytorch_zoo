import torch
import read_data
from torch.autograd import Variable
from Alexnet import AlexNet
import numpy as np
alex_net = torch.load('alexnet.pkl')
correct = 0
total = 0
for images, labels in read_data.test_loader:
    images, labels = Variable(images.cuda()), Variable(labels.cuda())
    outputs = alex_net(images)
    _, predicts = torch.max(outputs.data, 1)
    correct += torch.sum(predicts.cpu() == labels.cpu())
    total += labels.size(0)

print('acc:' + str(correct / total))

