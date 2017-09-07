## LeNet

use the Lenet to analysis the cifar-10 dataset
test accuracy: 60.15%

structure:

    (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  
    (max_pool_1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  
    (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  
    (max_pool_2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  
    (fc1): Linear (400 -> 120)
  
    (fc2): Linear (120 -> 84)
  
    (fc3): Linear (84 -> 10)
 
loss function: cross entropy
optimizer: SGD lr = 0.01, mometum = 0.9

loss graph:
![loss graph](https://github.com/yt4766269/pytorch_zoo/blob/master/LeNet/out.png)
