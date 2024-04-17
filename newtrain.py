import numpy as np
import os
import random
import shutil
import time
import warnings
from collections import defaultdict
from functools import reduce
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model import Detector
from data_reader.dataset_v1 import SpoofDatsetSystemID

from local import datafiles, trainer, validate, optimizer

import argparse
from adydata import dataaa
torch.mps.set_per_process_memory_fraction(0.0)
run_id="PA_trains_50"
best_acc1 = 0
batch_size=32
test_batch_size = 64
epochs = 20
start_epoch = 1
n_warmup_steps = 1000
log_interval = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
random_seed=0
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.mps.deterministic = True
torch.backends.mps.benchmark = False
    
np.random.seed(random_seed)
random.seed(random_seed)

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# model is trained for binary classification (for datalaoder) 
binary_class = True 


# create model
# kwargs = {'num_workers': 2, 'pin_memory': True} if device == torch.device('mps') else {}
model = Detector(9,2).to(device) 
num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('===> Model total parameter: {}'.format(num_model_params))

# Wrap model for multi-GPUs, if necessary
# if device == torch.device('mps') and torch.mps.device_count() > 1:
#     print('multi-gpu') 
#     model = nn.DataParallel(model).mps()

# define loss function (criterion) and optimizer
optim = optimizer.ScheduledOptim(
        torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, lr=3e-4, amsgrad=True),n_warmup_steps)

# optionally resume from a checkpoint
# if pretrained:
#     if os.path.isfile(pretrained):
#         print("===> loading checkpoint '{}'".format(pretrained))
#         checkpoint = torch.load(pretrained)
#         start_epoch = checkpoint['epoch']
#         best_acc1 = checkpoint['best_acc1']
#         model.load_state_dict(checkpoint['state_dict'])
#         optim.load_state_dict(checkpoint['optimizer'])
#         print("===> loaded checkpoint '{}' (epoch {})".format(pretrained, checkpoint['epoch']))
#     else:
#         print("===> no checkpoint found at '{}'".format(pretrained))

# Data loading code
train_data = dataaa(typee='PA',e='trains')
val_data = dataaa(typee='PA',e='dev')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=True)

best_epoch = 0
early_stopping, max_patience = 0, 100 # for early stopping
snapdir="model_snapshots/" + run_id
os.makedirs(snapdir[:-1], exist_ok=True) 
for epoch in range(start_epoch, start_epoch+epochs):
    #train
    running_loss = 0.0
    batch_time = optimizer.AverageMeter()
    data_time = optimizer.AverageMeter()
    losses = optimizer.AverageMeter()
    top1 = optimizer.AverageMeter()
    model.train()
    end = time.time()
    correct_val=0
    tot=0
    for i,(images,labels) in enumerate(train_loader):
        tot+=labels.size(0)
        # images=images.to(dtype=torch.float32)
        images=images.to(dtype=torch.float32)
        images=images.unsqueeze(1)
        images=images.to(device, non_blocking=True)
        labels=labels.to(device, non_blocking=True).view((-1,))
        outputs = model(images)
        loss = F.nll_loss(outputs, labels)
        predicted_classes = torch.argmax(outputs, dim=1)
        correct_val += (predicted_classes == labels).sum().item()
        losses.update(loss.item(), images.size(0))
        # compute gradient and do SGD step
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr = optim.update_learning_rate()
        print("Batch:{i}\n Loss{loss}",i=i,loss=loss)
        # if i % log_interval == 0:
    accs=correct_val/tot
    f=open(snapdir+"accs.txt",'a')
    f.write('\n'+"Epoch:"+str(epoch)+"accs:"+str(accs)+"loss:"+str(losses))
    f.close()
    print("ACCS",accs)
    print('Epoch: [{0}][{1}/{2}]\t'
      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
      'LR {lr:.6f}\t'
      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
       epoch, i, len(train_loader), batch_time=batch_time,
       data_time=data_time, lr=lr, loss=losses, top1=top1))

    # trainer.train(train_loader, model, optim, epoch, device, log_interval)
    #acc1 = validate.validate(val_loader, data_files['dev_utt2systemID'], model, device, log_interval)    
    #validate
    batch_time = optimizer.AverageMeter()
    losses = optimizer.AverageMeter()
    top1 = optimizer.AverageMeter()
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        # Iterate over the test dataset
        for i,(images,labels) in enumerate(val_loader):
            # Forward pass through the model
            # images=images.to(dtype=torch.float32)
            images=images.to(dtype=torch.float32)
            images=images.unsqueeze(1)
            images=images.to(device, non_blocking=True)
            labels=labels.to(device, non_blocking=True).view((-1,))
        
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)
            acc1, = optimizer.accuracy(outputs, labels, topk=(1, ))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % log_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

            # Get predicted labels
            _, predicted = torch.max(outputs, 1)
            # Update total count
            total += labels.size(0)
            # Update correct count
            correct += (predicted == labels).sum().item()
    accs2=correct/total
    f=open(snapdir+"accs2.txt",'a')
    f.write('\n'+"Epoch:"+str(epoch)+"accs2:"+str(accs2)+"loss:"+str(losses))
    f.close()
    print("ACC2",accs2)
    print('===> Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    acc2=top1.avg
    is_best=False
    if acc2>best_acc1:
        is_best=True
    best_acc1 = max(acc2, best_acc1)
    # adjust learning rate + early stopping 
    # if is_best:
    #     early_stopping = 0
    #     best_epoch = epoch + 1
    # else:
    #     early_stopping += 1
    #     if epoch - best_epoch > 2:
    #         optim.increase_delta()
    #         best_epoch = epoch + 1
    # if early_stopping == max_patience:
    #     break
    
    # save model
    optimizer.save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optim.state_dict(),
        }, is_best,  "model_snapshots/PA50/" + str(run_id), str(epoch) + ('_%.3f'%acc2) + ".pth.tar")


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run-id', action='store', type=str, default='0')
#     parser.add_argument('--data-feats', action='store', type=str, default='pa_spec')
#     parser.add_argument('--pretrained', action='store', type=str, default=None)
#     parser.add_argument('--configfile', action='store', type=str)
#     parser.add_argument('--random-seed', action='store', type=int, default=0)
#     args = parser.parse_args()

#     run_id = args.run_id
#     pretrained = args.pretrained
#     random_seed = args.random_seed


#     with open(args.configfile) as json_file:
#         config = json.load(json_file)

#     print(config)

#     data_files = datafiles.data_prepare[args.data_feats]
#     # model_params = config['model_params']
#     model_params = (0,2)
#     training_params = config['training_params']
    


#     ''' 
#     print(run_id)
#     print(pretrained)
#     print(data_files)
#     print(model_params)
#     print(training_params)
#     print(device)
#     exit(0)
#     '''
#     main(run_id, pretrained, data_files, model_params, training_params, device)

