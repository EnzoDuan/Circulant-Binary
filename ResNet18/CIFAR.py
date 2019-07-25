from __future__ import division
import os
import time
import argparse
import torch
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from utils import accuracy, AverageMeter, save_checkpoint, visualize_graph, get_parameters_size, RandomCrop
from tensorboardX import SummaryWriter
from resnet18_cifar import resnet18
from torch.backends import cudnn


parser = argparse.ArgumentParser(description='PyTorch GCN CIFAR Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--gpu', default=-1, type=int,
                    metavar='N', help='GPU device ID (default: -1)')
parser.add_argument('--dataset', default='CIFAR10', type=str, metavar='dataset', choices=['CIFAR10', 'CIFAR100'],
                    help='CIFAR10 or CIFAR100 (default: CIFAR10)')
parser.add_argument('--dataset_dir', default='/home/lcl/CIFAR', type=str, metavar='PATH',
                    help='path to dataset (default: ../../CIFAR)')
parser.add_argument('--comment', default='', type=str, metavar='INFO',
                    help='Extra description for tensorboard')
parser.add_argument('--dropout-rate', default=0, type=float,
                    metavar='N', help='dropout rate (default: 0)')
parser.add_argument('--nchannel', default=4, type=int,
                    metavar='N', help='nChannel (default: 4)')
parser.add_argument('--debug', default=False, action='store_true',
                    help='debug or not')

args = parser.parse_args()

use_cuda = (args.gpu >= 0) and torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")
if use_cuda:
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

best_prec1 = 0
writer_comment = '_'.join(['', 'ResNet18', 'lr', str(args.lr), 'M', str(args.nchannel), 'dropout', str(args.dropout_rate)])
if args.comment != '': writer_comment = '_'.join([writer_comment, args.comment])
print(writer_comment)
writer = SummaryWriter(comment=writer_comment)
iteration = 0


# Prepare the CIFAR dataset
if args.dataset == 'CIFAR10':
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        RandomCrop(32, padding=4, mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True,
                        download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False,
                        download=True,transform=test_transform)
    num_classes = 10
else:
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                    std=[0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR100(root=args.dataset_dir, train=True,
                        download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=args.dataset_dir, train=False,
                        download=True,transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, shuffle=True)

# Load model
model = resnet18(M=args.nchannel, method='GConv').to(device)
print(model)

# Try to visulize the model
try:
	visualize_graph(model, writer, input_size=(1, 3, 32, 32))
except:
	print('\nNetwork Visualization Failed! But the training procedure continue.')

# Calculate the total parameters of the model
print('Model size: {:0.2f} million float parameters'.format(get_parameters_size(model)/1e6))

MFilter_params = [param for name, param in model.named_parameters() if name[-8:] == 'MFilters']
Other_params = [param for name, param in model.named_parameters() if name[-8:] != 'MFilters']
# optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=3e-05)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-05)
optimizer1 = optim.SGD([{'params': MFilter_params, 'weight_decay': 0, 'lr': args.lr * 0.1, 'momentum':0},
                        {'params': Other_params}], lr=args.lr, momentum=args.momentum, weight_decay=5e-04, nesterov=True) # ????
scheduler = MultiStepLR(optimizer1, milestones=[60,120,160], gamma=0.2)

_params = [param for name, param in model.named_parameters() if 'conv' not in name]
# print _params
optimizer2 = torch.optim.SGD([{'params':_params}], lr=args.lr, momentum=args.momentum, weight_decay=5e-04, nesterov=True)
#scheduler = MultiStepLR(optimizer2, milestones=[60,120,160], gamma=0.2)
criterion = nn.CrossEntropyLoss().to(device)

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

def train(epoch):
    model.train()
    global iteration
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration += 1
        data, target = data.to(device), target.to(device)
        optimizer1.zero_grad()
        output = model(data)
        prec1, = accuracy(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer1.step()
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), prec1))
            writer.add_scalar('Loss/Train', loss.item(), iteration)
            writer.add_scalar('Accuracy/Train', prec1, iteration)
    epoch_time = time.time() - st
    print('Epoch time:{:0.2f}s'.format(epoch_time))
    if args.debug:
        for name, param in model.named_parameters():
            bins = 'doane'
            if 'bn' in name: # skip bn parameters
                continue
            elif 'MFilters'in name:
                MFilter_size = param.size(2)
                bins = args.nchannel * MFilter_size * MFilter_size
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch, bins=bins)
    scheduler.step()
def finetune(epoch):

    # before = model.state_dict()['conv4.weight']
    model.train()
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer2.zero_grad()
        output = model(data)
        prec1, = accuracy(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer2.step()
       # if batch_idx % args.print_freq == 0:
       #     print('Finetune [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}'.format(
       #         batch_idx * len(data), len(train_loader.dataset),
       #         100. * batch_idx / len(train_loader), loss.item(), prec1))
            # writer.add_scalar('Loss/Train', loss.item(), iteration)
            # writer.add_scalar('Accuracy/Train', prec1, iteration)
    epoch_time = time.time() - st
    # after = model.state_dict()['conv4.weight']
    # print torch.equal(before, after)
    print('Epoch time:{:0.2f}s'.format(epoch_time))


def test(epoch):
    model.eval()
    test_loss = AverageMeter()
    acc = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss.update(F.cross_entropy(output, target, size_average=True).item(), target.size(0))
            prec1, = accuracy(output, target) # test precison in one batch
            acc.update(prec1, target.size(0))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss.avg, acc.avg))
    writer.add_scalar('Loss/Test', test_loss.avg, epoch)
    writer.add_scalar('Accuracy/Test', acc.avg, epoch)
    return acc.avg

for epoch in range(args.start_epoch, args.epochs):
    print('------------------------------------------------------------------------')
    train(epoch+1)
    #finetune(epoch+1)
    print('---------------------------------------------------------------------------')
    prec1 = test(epoch+1)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer1.state_dict(),
    }, is_best)
    writer.add_scalar('Best TOP1/For now', best_prec1, 0)

print('Finished!')
print('Best Test Precision@top1:{:.2f}'.format(best_prec1))
writer.add_scalar('Best TOP1/Final', best_prec1, 0)
writer.close()
