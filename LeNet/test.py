import torch
import read_image
from torch.autograd import Variable
net = torch.load('model.pkl')

correct = 0
total = 0
for data in read_image.testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    for i in range(0, len(labels)):
        total += 1
        if labels[i] == predicted[i]:
            correct +=1

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))