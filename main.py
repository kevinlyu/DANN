import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import numpy as np
import model
import dataloader

# data loader
target_loader = torch.utils.data.DataLoader(dataloader.MNISTM(
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=100, shuffle=True)

source_loader = torch.utils.data.DataLoader(datasets.MNIST(
    "../dataset/mnist/", train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=100, shuffle=True)


# network components
feature_extractor = model.Extractor()
class_classifier = model.Classifier()
domain_discriminator = model.Discriminator()


# train the model on GPU
feature_extractor.cuda()
class_classifier.cuda()
domain_discriminator.cuda()

# training criterion
# with the help  gradient reversal layer, it can perform min-max game using the same criterion
class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()

# optimizer
optimizer = optim.SGD([{"params": feature_extractor.parameters()},
                       {"params": class_classifier.parameters()},
                       {"params": domain_discriminator.parameters()}], lr=0.01, momentum=0.9)


for epoch in range(1):
    print("Epoch {}".format(epoch))

    
    # steps 
    start_steps = epoch*len(source_loader)
    total_steps = epoch*len(source_loader)
    
    for index, (source, target) in enumerate(zip(source_loader, target_loader)):

        # hyper parameters
        p = float(index, + start_steps)/total_steps # for utils.optimizer.scheduler to adjust learning dunamically
        constant = 2.0 / (1.0+np.exp(-10*p))-1

        source_data, source_label = source
        target_data, target_label = target

        # move data to GPU
        source_data, source_label = Variable(source_data.cuda()), Variable(source_label.cuda())
        target_data, targe_label = Variable(target_data.cuda()), Variable(target_label.cuda())


        # move domain labels to GPU
        source_labels = Variable(torch.zeros((source_data.size()[0])).type(torch.LongTensor).cuda()) 
        target_labels = Variable(torch.zeros((target_data.size()[0])).type(torch.LongTensor).cuda()) 


        # extract features from domains respectively
        source_z = feature_extractor(source_data)
        target_z = feature_extractor(target_data)

        # classification loss of task classifier
        class_pred = class_classifier(source_z)
        class_loss = class_criterion(class_pred, source_label)

        # classification loss of domain discriminator
        source_pred = domain_discriminator(source_z, constant)
        target_pred = domain_discriminator(target_z, constant)
        source_loss = domain_criterion(source_pred, source_labels)
        target_loss = domain_criterion(target_pred, target_labels)
        domain_loss = source_loss + target_loss

        # total loss
        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()

        if(index+1) % 10 == 0:
            print("")