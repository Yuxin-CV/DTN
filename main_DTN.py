import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import models_resnet as model_e
import generator as model_g
import task_generator as loader
import numpy as np
import random
import math
import scipy.stats as stats

from utils import  AverageMeter, accuracy, mkdir_p


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./miniImageNet/',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=31, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=0, type=int,
                    metavar='E', help='evaluate model on validation set')
parser.add_argument('-g', '--gen-num', default=64, type=int,
                    metavar='G', help='mini-batch size (default: 64)')
parser.add_argument('--N-way', default=5, type=int,
                    metavar='NWAY', help='N_way (default: 5)')
parser.add_argument('--N-shot', default=5, type=int,
                    metavar='NSHOT', help='N_shot (default: 1)')
parser.add_argument('--N-query', default=15, type=int,
                    metavar='NQUERY', help='N_query (default: 15)')

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main():
    global args
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    cudnn.benchmark = True

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    model_E = model_e.Net().cuda()
    model_G = model_g.GeneratorNet().cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([
        {'params': model_E.parameters()},
        {'params': model_G.parameters(), 'lr': args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,25,30,40,45,50,55,60,65,80,100,120,140], gamma=0.4)


    # Data loading code
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

    train_aug_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),

            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_train=True)

    base_train_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    gen_support_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.gen_num*2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler = loader.GeneratorSupportSampler(n_class=80, n_support_pairs=args.gen_num))

    gen_train_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.N_way*(args.N_query+args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler = loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query+args.N_shot, n_class=80))

    gen_support_for_test_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.gen_num * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSupportSampler(n_class=80, n_support_pairs=args.gen_num))

    test_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((224, 224)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_test=True)

    gen_test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5 * (args.N_query + args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSampler(num_of_class=5, num_per_class=args.N_query + args.N_shot, n_class=20))

    if args.evaluate:

        if args.resume:
            print('==> Resuming from generator checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.resume)
            model_G.load_state_dict(checkpoint['state_dict_G'])
            model_E.load_state_dict(checkpoint['state_dict_E'])

        losses = AverageMeter()
        top1 = AverageMeter()
        H = AverageMeter()


        for i in range(10):
            val_loss, val_acc, h = validate(gen_support_for_test_loader, gen_test_loader, model_E, model_G, criterion)
            losses.update(val_loss)
            top1.update(val_acc)
            H.update(h)

            print(i, losses.avg, top1.avg, H.avg)

        return 0

    train_acc, train_loss, base_train_loss, base_train_acc = 0, 0, 0, 0
    best_val_acc = 0

    # There is a little inconsistency between the code and the paper: 
    # we have an additional auxiliary epoch A in the actual implementation.
    # In the paper, the the whole training sequence is: S = 10A + (4A + 1M) × 2 + (3A + 2M) × 2
    # In the code,  the the whole training sequence is: S = 11A + (4A + 1M) × 2 + (3A + 2M) × 2
    # The reason is that there is a problem with the setting of milestones of the training schedule in line 84. 
    TRAIN_PHASE = ['a']*11 + ['a']*4 + ['m']*1 + ['a']*4 + ['m']*1 + (['a']*3 + ['m']*2)*20

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        torch.cuda.empty_cache()

        
        train_phase = TRAIN_PHASE[epoch]

        
        if train_phase == 'a':
            print('phase: base_train...')
            base_train_loss, base_train_acc = base_train(base_train_loader, model_E, criterion, optimizer, epoch)

            # writer.add_scalars('loss', {"base_train_loss":base_train_loss}, epoch)
            # writer.add_scalars('acc', {"base_train_acc":base_train_acc}, epoch)
        else:
            print('phase: meta_train...')
            train_loss, train_acc = train(gen_support_loader, gen_train_loader, model_E, model_G, criterion, optimizer, epoch)

            # writer.add_scalars('loss', {"meta_train_loss":train_loss}, epoch)
            # writer.add_scalars('acc', {"mata_train_acc":train_acc}, epoch)

        # writer.add_scalars('lr', {'lr_E':optimizer.param_groups[0]['lr'], 'lr_G':optimizer.param_groups[1]['lr']}, epoch)
        
        print('meta_train_acc:',train_acc)
        print('base_train_acc:',base_train_acc)


    # save the final checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict_E': model_E.state_dict(),
        'state_dict_G': model_G.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, epoch, checkpoint=args.checkpoint)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def train(gen_support_loader, gen_train_loader, model_E, model_G, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model_E.train()
    model_G.train()
        # 冻结特征提取的网络
    # for param in model_E.parameters():
    #   param.requires_grad = False

    for inter in range(400):
        input, target = gen_train_loader.__iter__().next()
        gen_support_input, gen_support_target = gen_support_loader.__iter__().next()

        input = input.cuda()
        gen_support_input = gen_support_input.cuda()
        support_input = input.view(args.N_way, args.N_query + args.N_shot, 3, 224, 224)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 224, 224)
        query_input   = input.view(args.N_way, args.N_query + args.N_shot, 3, 224, 224)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 224, 224)
        gen_support_1 = gen_support_input.view(-1, 2, 3, 224, 224)[:,0,:,:,:]
        gen_support_2 = gen_support_input.view(-1, 2, 3, 224, 224)[:,1,:,:,:]

        weight = torch.zeros((args.N_way, 1024), requires_grad=True).cuda()
        support_input, _ = model_E(support_input)
        gen_support_1, _ = model_E(gen_support_1)
        gen_support_2, _ = model_E(gen_support_2)
        query_input  , _ = model_E(query_input)

        for i in range(args.N_way):
            weight_point = torch.zeros(args.N_shot*(args.gen_num+1), 1024)
            for j in range(args.N_shot):
                gen_feature, _ = model_G(gen_support_1, gen_support_2, support_input[i*args.N_shot+j])
                features = torch.cat((gen_feature, support_input[i*args.N_shot+j].unsqueeze(0)), 0)
                weight_point[j*(args.gen_num+1):(j+1)*(args.gen_num+1)] = features
            weight[i] = torch.mean(weight_point, 0)
        weight = model_E.l2_norm(weight)

        # query_input = model_E.l2_norm(query_input)

        predict = torch.matmul(query_input, torch.transpose(weight,0,1))*model_G.s
        gt = np.tile(range(args.N_way), args.N_query)
        gt.sort()
        gt = torch.cuda.LongTensor(gt)

        acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
        loss = criterion(predict, gt)

        losses.update(loss.item(), predict.size(0))
        top1.update(acc.item(), predict.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (inter+1)%200==0:
            print('meta_train:', inter+1, 'loss:', losses.avg, 'acc:', top1.avg)

    return (losses.avg, top1.avg)




def validate(gen_support_for_test_loader, gen_test_loader, model_E, model_G, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model_E.eval()
    model_G.eval()

    with torch.no_grad():
        accuracies = []
        for inter in range(600):
            input, target = gen_test_loader.__iter__().next()
            gen_support_input, gen_support_target = gen_support_for_test_loader.__iter__().next()
            support_input = input.view(5,args.N_query+args.N_shot,3,224,224)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 224, 224).cuda()
            query_input = input.view(5,args.N_query+args.N_shot,3,224,224)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 224, 224).cuda()
            gen_support_1 = gen_support_input.view(-1, 2, 3, 224, 224)[:,0,:,:,:].cuda()
            gen_support_2 = gen_support_input.view(-1, 2, 3, 224, 224)[:,1,:,:,:].cuda()

            weight = torch.zeros((5, 1024), requires_grad=True).cuda()
            support_input, _ = model_E(support_input)
            gen_support_1, _ = model_E(gen_support_1)
            gen_support_2, _ = model_E(gen_support_2)
            query_input, _ = model_E(query_input)

            R = 1

            for i in range(5):
                weight_point = torch.zeros(args.N_shot*(args.gen_num+1), 1024)
                for j in range(args.N_shot):
                    spt_input = support_input[i*args.N_shot+j]
                    for r in range(R):
                        gen_feature, _ = model_G(gen_support_1, gen_support_2, spt_input)
                        features = torch.cat((gen_feature, spt_input.unsqueeze(0)), 0)
                        spt_input = torch.mean(features, 0)
                    weight_point[j*(args.gen_num+1):(j+1)*(args.gen_num+1)] = features
                weight[i] = torch.mean(weight_point, 0)

            weight = model_E.l2_norm(weight)

            # query_input = model_E.l2_norm(query_input)

            predict = torch.matmul(query_input, torch.transpose(weight,0,1))*model_G.s
            gt = np.tile(range(5), args.N_query)
            gt.sort()
            gt = torch.cuda.LongTensor(gt)

            acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
            accuracies.append(acc.item())
            loss = criterion(predict, gt)

            losses.update(loss.item(), predict.size(0))
            top1.update(acc.item(), predict.size(0))


            if (inter+1)%100==0:
                print('test:', inter+1, 'loss:', losses.avg, 'acc:', top1.avg)


    mean, h = mean_confidence_interval(accuracies)
    return (losses.avg, top1.avg, h)



def base_train(base_train_loader, model_E, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model_E.train()
    
    # for param in model_E.parameters():
    #   param.requires_grad = True

    for batch_idx, (input, target) in enumerate(base_train_loader):
        # print(target)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        _, output = model_E(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_E.weight_norm()
        if (batch_idx+1)%250==0:
            print('base_train:', batch_idx+1, 'loss:', losses.avg, 'acc:', top1.avg)

    return (losses.avg, top1.avg)

def base_val(base_train_loader, model_E, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model_E.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(base_train_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            _, output = model_E(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            model_E.weight_norm()
            if (batch_idx+1)%250==0:
                print('base_test:', batch_idx+1, 'loss:', losses.avg, 'acc:', top1.avg)

    return (losses.avg, top1.avg)


def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    torch.save(state, filepath)
    print('save checkpoint success', epoch)


if __name__ == '__main__':
    main()
