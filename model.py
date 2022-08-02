import torch

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()    #继承torch.nn.Module的属性
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2,stride=2)

        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84,10)


    def forward(self, x):
        batch_size = x.size(0)
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=x.view(-1,4*4*16)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

