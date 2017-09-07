from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
import read_image

from Net import Net
from CONSTANT import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, EPOCH, CALCULATE_LOSS

net = Net()
net.cuda()


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=net.parameters(),lr = 1e-2, momentum=0.9)
plt.figure(num=1)
plt_x = []
plt_y = []

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(read_image.trainloader):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]


        if i % CALCULATE_LOSS == CALCULATE_LOSS - 1:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / CALCULATE_LOSS))
            plt_x.append(50000 / TRAIN_BATCH_SIZE * epoch + i)
            plt_y.append(running_loss)
            running_loss = 0.0

print("finish training")

torch.save(net, 'model.pkl')
plt.plot(plt_x, plt_y)
plt.savefig('out.png')