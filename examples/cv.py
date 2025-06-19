from GeN import setup, lr_parabola
from prodigyopt import Prodigy
from dadaptation import DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptLion, DAdaptSGD
from dog import DoG, PolynomialDecayAverager

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import timm
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from time import time
import numpy as np



def main(args):
    device= torch.device("cuda:0")

    # Data
    print('==> Preparing data..')

    train_transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    test_transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    if args.dataset_name in ['SVHN','Food101','Flowers102','GTSRB']:
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='train', download=True, transform=train_transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='test', download=True, transform=test_transformation)
    elif args.dataset_name in ['CIFAR10','CIFAR100']:
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', train=True, download=True, transform=train_transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', train=False, download=True, transform=test_transformation)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)
    
    n_acc_steps = args.bs // args.mini_bs # gradient accumulation steps

    # Model
    if args.dataset_name in ['SVHN','CIFAR10']:
        num_classes=10
    elif args.dataset_name in ['CIFAR100']:
        num_classes=100
    elif args.dataset_name in ['Food101']:
        num_classes=101
    elif args.dataset_name in ['GTSRB']:
        num_classes=43


    # Model
    print('==> Building ', args.model, '..gradient accumulation steps..', n_acc_steps)
    net = timm.create_model(args.model,pretrained=True,num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()


    if args.optim=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim=='adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=='prodigy':
        optimizer = Prodigy(net.parameters(), weight_decay=args.weight_decay, decouple=args.decouple)
    elif args.optim=='dadaptadam':
        optimizer = DAdaptAdam(net.parameters(), weight_decay=args.weight_decay, decouple=True) #lr=1.0, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, 
        # log_every=0,decouple=False,use_bias_correction=False,d0=1e-6, growth_rate=float('inf'),fsdp_in_use=False
    elif args.optim=='dadaptsgd':
        optimizer = DAdaptSGD(net.parameters(), weight_decay=args.weight_decay) #lr=1.0, momentum=0.0, weight_decay=0, 
        # log_every=0, d0=1e-6, growth_rate=float('inf'), fsdp_in_use=False
    elif args.optim=='dog':
        optimizer = DoG(net.parameters(), weight_decay=args.weight_decay)
        averager = PolynomialDecayAverager(net)                   
    else:
        print('optimizer does not exist!!!')

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    else:
        print('No lr scheduler...')


    def train(epoch):
        tr_iter=iter(trainloader)

        net.train()
        train_loss, correct, total = 0,0,0
        total_tr = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            total_tr += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss/=n_acc_steps
            loss.backward()

                
            if (batch_idx+1)%n_acc_steps==0:
                if ((batch_idx+1)/n_acc_steps)%args.lazy_freq==0:
                    if args.lr_scheduler == 'GeN':
                        scale_list = np.linspace(1,0,args.epochs+1)[:-1]
                        lr_parabola(net, optimizer, criterion, n_acc_steps, tr_iter= tr_iter,  task='image_cls', scale=scale_list[epoch])
    
                optimizer.step()
                if args.optim == 'dog' or args.optim == 'ldog':
                    averager.step()
                optimizer.zero_grad()
            
                if (batch_idx+1)%(len(trainloader)//50)==0:
                    print('Epoch: ', epoch, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                    % (train_loss/n_acc_steps, 100.*correct/total, correct, total))
                    test(epoch, full_data = False)
                    net.train()
                train_loss, correct, total = 0,0,0


    def test(epoch, full_data=True):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.mean().item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if full_data==False and batch_idx>=5:
                    break

            print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

    start_time = time()
    for epoch in range(args.epochs):
        train(epoch)
        if args.lr_scheduler in ['cosine','multistep']:
            scheduler.step()
    print(f"Total training time for {args.epochs} epochs = {time() - start_time}s")
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CV Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=5, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=500, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',help='https://pytorch.org/vision/stable/datasets.html')
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, help="choose from 'vit_tiny_patch16_224', 'vit_large_patch16_224', 'crossvit_small_240', 'alexnet' ")
    parser.add_argument('--lazy_freq', default=4, type=int, help='lr update frequency')
    parser.add_argument('--optim', default='adamw',  type=str)
    parser.add_argument('--lr_scheduler', default='none',  type=str)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    setup(args.seed)
    print(vars(args))
    main(args)
