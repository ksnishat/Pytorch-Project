import torch
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
    def __init__(self,input_channels, out_channels, stride):
        super(ResBlock,self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.stride = stride
        self.input_channels = input_channels
        # delare layer Conv, BN in Resblock
        self.conv1 = torch.nn.Conv2d(in_channels=self.input_channels, out_channels= out_channels, stride=stride, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        if self.stride != 1 or self.input_channels != self.out_channels:
            self.reformat_input = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_channels, out_channels=self.out_channels, stride=self.stride, kernel_size=1),
                torch.nn.BatchNorm2d(self.out_channels))
        else:
            self.reformat_input = torch.nn.BatchNorm2d(self.out_channels)

    def forward(self, input):
        output =self.conv1(input)
        output =self.bn1(output)
        output = F.relu(output)
        output =self.conv2(output)
        output =self.bn2(output)
        output +=self.reformat_input(input)
        output = F.relu(output)

        return output


class ResNet18(torch.nn.Module):

    def __init__(self, input_channels=3, classes=2):
        super(ResNet18, self).__init__()
        self.input_channels = input_channels
        self.classes = classes
        # Declare layer in ResNet18
        self.conv = torch.nn.Conv2d(in_channels=self.input_channels, out_channels= 64, stride=2, kernel_size=7, padding=1)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.resblock1 = ResBlock(input_channels=64, out_channels= 64,stride=1)
        self.resblock2 = ResBlock(input_channels=64, out_channels=128,stride=2)
        self.resblock3 = ResBlock(input_channels = 128, out_channels= 256,stride=2)
        self.resblock4 = ResBlock(input_channels= 256, out_channels=512, stride=2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = torch.nn.Linear(in_features=512, out_features= self.classes)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.max_pooling(output)
        output = self.resblock1(output)
        output = self.resblock2(output)
        output = self.resblock3(output)
        output = self.resblock4(output)
        output = self.avg_pool(output)
        # flatten
        output = torch.flatten(output, start_dim=1)
        # output = F.dropout(output) # data go through Fully connected
        output = self.fc(output)
        return output

    




















