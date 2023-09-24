"""This file trains the neural network using the supervised data acquired by the
scaper file from the API. """

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import scraper
import text_dataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(250, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.soft = nn.Softmax(1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.25)         #Dropout layers help train the network on partial input, as it will be for the majority of use cases

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out += residual
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.soft(out)

        return out


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(output, target)
            test_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (correct/ len(test_loader.dataset))


def main(lr, bs, gm):
    # Training settings

    epochs = 7
    learnrate = lr
    batchsize = bs
    gamma = gm
    interval = 100
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=batchsize, metavar='N',
                        help='input batch size for training (default: 16')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: 6)')
    parser.add_argument('--lr', type=float, default=learnrate, metavar='LR',
                        help='learning rate (default: 2.0)')
    parser.add_argument('--gamma', type=float, default=gamma, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=interval, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    traintxt, trainlabels = 'raw/trainvalues.txt','raw/trainlabels.txt'
    testtxt, testlabels = 'raw/testvalues.txt','raw/testlabels.txt'
    
    
    traindataset = text_dataset.CustomTextDataset(traintxt, trainlabels, transforms.ToTensor()) #Load the data using the custom dataset object
    testdataset = text_dataset.CustomTextDataset(testtxt, testlabels, transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(traindataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testdataset, **train_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    accuracy = [0 for i in range(0,epochs)]
    i = 0
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy[i] = test(model, device, test_loader)
        scheduler.step()
        i+=1

    torch.save(model.state_dict(), "pytorch_model.pt")

    return accuracy     #Returns the accuracy of the network after each epoch for logging purposes

if __name__ == '__main__':
    main(1,8,0.9)
