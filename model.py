import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    '''
    Gradient Reversal Layer
    '''
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None

    # pylint raise E0213 warning here
    def grad_reverse(x, constant):
        '''
        Extension of grad reverse layer
        '''
        return GradReverse.apply(x, constant)


class Extractor(nn.Module):

    '''
    Shared feature extractor
    '''

    def __init__(self):
        super(Extractor, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5))
        self.layer.add_module("bn1", nn.BatchNorm2d(64))
        self.layer.add_module("pool1", nn.MaxPool2d(2))
        self.layer.add_module("conv2", nn.Conv2d(64, 50, kernel_size=5))
        self.layer.add_module("bn2", nn.BatchNorm2d(50))
        self.layer.add_module("conv2_drop", nn.Dropout2d())
        self.layer.add_module("pool2", nn.MaxPool2d(2))

    def forward(self, x):
        x = x.view(-1, 50*4*4)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(50*4*4, 100)
        self.bn1 = nn.BatchNorm2d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm2d(100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(self.self.bn2(logits))
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)


class Discriminator(nn.Module):
    '''
    Domain classifier
    '''

    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(50*4*4, 100)
        self.bn1 = nn.BatchNorm2d(100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x, constant):
        '''
        constant here is the variable lambda in DANN paper
        '''
        x = GradReverse(x, constant)
        logits = F.relu(self.bn1(self.fc1(x)))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits
