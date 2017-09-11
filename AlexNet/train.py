import Alexnet
import read_data
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch

alex_net = Alexnet.AlexNet()
alex_net.cuda()
criterion = CrossEntropyLoss()
optimizer = optim.SGD(params=alex_net.parameters(), lr = 1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
for images, labels in read_data.train_loader:
    images, labels = Variable(images.cuda()), Variable(labels.cuda())
    optimizer.zero_grad()
    outputs = alex_net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('loss:' + loss.data[0])

if __name__ == '__main__':
    torch.save(alex_net, 'alexnet.pkl')