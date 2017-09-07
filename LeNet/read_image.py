import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CONSTANT import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, DATA_PATH

transform = transforms.Compose(
    transforms=[transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=DATA_PATH + 'cifar-10-batches-py', train=True, download=True, transform=transform)
# trainset = torchvision.datasets.MNIST(root = './MNIST', train = True, download= True, transform  = transform)
trainloader = DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH + 'cifar-10-batches-py', train=False, download=True, transform=transform)
# testset = torchvision.datasets.MNIST(root='./MNIST', train = True, download=True, transform = transform)
testloader = DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
