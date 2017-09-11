from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader

trans = transforms.Compose(transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='cifar10', train=True, transform=trans, download=True)
test_dataset = datasets.CIFAR10(root='cifar10', train=False, download=True, transform=trans)
train_loader = DataLoader(dataset = train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset = test_dataset, batch_size=4, shuffle=True, num_workers=4)