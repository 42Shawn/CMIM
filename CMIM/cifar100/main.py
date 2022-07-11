import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models_bnn
import models
import numpy as np
from torch.autograd import Variable
from utils.options import args
from utils.common import *
from modules import *
from datetime import datetime 
import dataset
from dataset import CIFAR100IdxSample
import torchvision.transforms as transforms

import cmim
from contrastive_alignment import contrastive_alignment

def main():
    global args, best_prec1, conv_modules
    best_prec1 = 0

    random.seed(args.seed)
    if args.evaluate:
        args.results_dir = './results'
    save_path = os.path.join(args.results_dir, args.dataset + args.model + args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.resume:
        with open(os.path.join(save_path,args.save+'config.txt'), 'w') as args_file:
            args_file.write(str(datetime.now())+'\n\n')
            for args_n,args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v,int) else args_v
                args_file.write(str(args_n)+':  '+str(args_v)+'\n')

        setup_logging(os.path.join(save_path, args.save + 'logger.log'))
        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)
    else: 
        setup_logging(os.path.join(save_path, 'logger.log'), filemode='a')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
    else:
        args.gpus = None

    if args.dataset=='tinyimagenet':
        num_classes=200
        model_zoo = 'models.'
    elif args.dataset=='imagenet':
        num_classes=1000
        model_zoo = 'models.'
    elif args.dataset=='cifar10': 
        num_classes=10
        model_zoo = 'models_bnn.'
    elif args.dataset=='cifar100': 
        num_classes=100
        model_zoo = 'models_bnn.'

    if len(args.gpus)==1:
        model = eval(model_zoo+args.model)(num_classes=num_classes).cuda()
        model_fp = eval(model_zoo+args.model_fp)(num_classes=num_classes).cuda()
    else: 
        model = nn.DataParallel(eval(model_zoo+args.model)(num_classes=num_classes))

    checkpoint_fp = torch.load(args.model_fp_dir)
    pretrained_dict = checkpoint_fp['state_dict']
    model_dict = model_fp.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model_fp.load_state_dict(model_dict)

    if not args.resume:
        logging.info("creating model %s", args.model)
        logging.info("model structure: %s", model)
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

    # evaluate
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
        else: 
            checkpoint = torch.load(args.evaluate)
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                        args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = os.path.join(save_path, args.model+'checkpoint.pth.tar')
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, args.model + 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            if len(args.gpus)>1:
                checkpoint['state_dict'] = dataset.add_module_fromdict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion = criterion.type(args.type)
    model = model.type(args.type)

    if args.evaluate:
        val_loader = dataset.load_data(
                    type='val',
                    dataset=args.dataset, 
                    data_path=args.data_path,
                    batch_size=args.batch_size, 
                    batch_size_test=args.batch_size_test, 
                    num_workers=args.workers)
        with torch.no_grad():
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, model_fp, criterion, 0)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    if args.dataset=='imagenet':
        train_loader = dataset.get_imagenet(
                        type='train',
                        image_dir=args.data_path,
                        batch_size=args.batch_size,
                        num_threads=args.workers,
                        crop=224,
                        device_id='cuda:0',
                        num_gpus=1)
        val_loader = dataset.get_imagenet(
                        type='val',
                        image_dir=args.data_path,
                        batch_size=args.batch_size_test,
                        num_threads=args.workers,
                        crop=224,
                        device_id='cuda:0',
                        num_gpus=1)

    elif args.dataset == 'cifar100':
        _, val_loader = dataset.load_data(
            dataset=args.dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            batch_size_test=args.batch_size_test,
            num_workers=args.workers)
        train_dataset = CIFAR100IdxSample
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        train_transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_loader = torch.utils.data.DataLoader(
            train_dataset(root=args.data_path,
                          transform=train_transform,
                          train=True,
                          download=False,
                          n=args.nce_n,
                          mode='exact'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
    else:
        _, val_loader = dataset.load_data(
                                    dataset=args.dataset,
                                    data_path=args.data_path,
                                    batch_size=args.batch_size,
                                    batch_size_test=args.batch_size_test,
                                    num_workers=args.workers)
        train_dataset = CIFAR10IdxSample
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        train_transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        train_loader = torch.utils.data.DataLoader(
            train_dataset(root=args.data_path,
                          transform=train_transform,
                          train=True,
                          download=False,
                          n=args.nce_n,
                          mode='exact'),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )


    contrastive_aligning = contrastive_alignment(512, 512, args.feat_dim, args.nce_n, args.nce_t, args.nce_mom, len(train_loader.dataset))

    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr':args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    def cosin(i,T,emin=0, emax=0.01):
        "customized cos-lr"
        return emin+(emax-emin)/2 * (1+np.cos(i*np.pi/T))

    if args.resume:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosin(args.start_epoch-args.warm_up*4, args.epochs-args.warm_up*4,0, args.lr)
    if args.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warm_up*4, eta_min = 0, last_epoch=args.start_epoch-args.warm_up*4)
    elif args.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1, last_epoch=-1)
    if not args.resume:
        logging.info("criterion: %s", criterion)
        logging.info('scheduler: %s', lr_scheduler)

    def cpt_tk(epoch):
        "compute t&k in back-propagation"
        T_min, T_max = torch.tensor(args.Tmin).float(), torch.tensor(args.Tmax).float()
        Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
        t = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
        k = max(1/t,torch.tensor(1.)).float()
        return t, k

    #* setup conv_modules.epoch
    conv_modules=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):
            conv_modules.append(module)

    for epoch in range(args.start_epoch+1, args.epochs):
        time_start = datetime.now()

        #*warm up
        if args.warm_up and epoch <5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch+1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])

        #* compute t/k in back-propagation
        t,k = cpt_tk(epoch)
        for name,module in model.named_modules():
            if isinstance(module,nn.Conv2d):
                module.k = k.cuda()
                module.t = t.cuda()
        for module in conv_modules:
            module.epoch = epoch
        # train
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, model_fp, criterion, contrastive_aligning, epoch, optimizer)

        #* adjust Lr
        if epoch >= 4 * args.warm_up:
            lr_scheduler.step()

        # evaluate 
        with torch.no_grad():
            for module in conv_modules:
                module.epoch = -1
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, model_fp, criterion, epoch)

        # remember best prec
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = max(val_prec1, best_prec1)
            best_epoch = epoch
            best_loss = val_loss

        # save model
        if epoch % 1 == 0:
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_parameters = model.module.parameters() if len(args.gpus) > 1 else model.parameters()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'parameters': list(model_parameters),
            }, is_best, path=save_path)

        if args.time_estimate > 0 and epoch % args.time_estimate==0:
           time_end = datetime.now()
           cost_time,finish_time = get_time(time_end-time_start,epoch,args.epochs)
           logging.info('Time cost: '+cost_time+'\t'
                        'Time of Finish: '+finish_time)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        print('best pred', best_prec1)


    logging.info('*'*50+'DONE'+'*'*50)
    logging.info('\n Best_Epoch: {0}\t'
                     'Best_Prec1 {prec1:.4f} \t'
                     'Best_Loss {loss:.3f} \t'
                     .format(best_epoch+1, prec1=best_prec1, loss=best_loss))

def forward(data_loader, model, model_fp, criterion, contrastive_aligning, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    d_model = cmim.cmim(model_fp, model, contrastive_aligning)

    for i, (img, target, idx, sample_idx) in enumerate(data_loader, start=1):
        # measure data loading time
        data_time.update(time.time() - end)
        if i==1 and training:
            for module in conv_modules:
                module.epoch=-1
        if args.gpus is not None:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            idx = idx.cuda(non_blocking=True)
            sample_idx = sample_idx.cuda(non_blocking=True)


        # print(img.shape)
        output = model(img)
        loss_cmim = d_model(img, idx, sample_idx)
        loss_ce = criterion(output, target)
        loss = loss_ce + args.alpha * loss_cmim

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))


        if training:
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses,
                             top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def forward_val(data_loader, model, model_fp, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for i, (inputs, target) in enumerate(data_loader, start=1):
        # measure data loading time
        data_time.update(time.time() - end)
        if i == 1 and training:
            for module in conv_modules:
                module.epoch = -1
        if args.gpus is not None:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        input_var = Variable(inputs.type(args.type))
        target_var = Variable(target)

        output = model(input_var)
        loss_ce = criterion(output, target_var)
        loss = loss_ce

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                batch_time=batch_time,
                data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, model_fp, criterion, contrastive_aligning, epoch, optimizer):
    model.train()
    return forward(data_loader, model, model_fp, criterion, contrastive_aligning, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, model_fp, criterion, epoch):
    model.eval()
    return forward_val(data_loader, model, model_fp, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
