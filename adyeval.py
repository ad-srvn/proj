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

from local import datafiles, prediction, trainer, validate, optimizer

import argparse
from adydata import dataaa
def main(data_files, model_params, training_params, device):
    """ forward pass dev and eval data to trained model  """
    batch_size = training_params['batch_size']
    batch_size=32
    test_batch_size = training_params['test_batch_size']
    epochs = training_params['epochs']
    start_epoch = training_params['start_epoch']
    n_warmup_steps = training_params['n_warmup_steps']
    log_interval = training_params['log_interval']

    kwargs = {'num_workers': 4, 'pin_memory': True} if device == torch.device('mps') else {}

    # create model
    model = Detector(0,2).to(device) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print('===> Model total parameter: {}'.format(num_params))
    checkpoint = torch.load('/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/model_snapshots/PAdiff/SEResNet34Debugfeats0/17_100.000.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    val_data = dataaa(typee='PA',e='trains')
    eval_data = dataaa(typee='LA',e='eval')
    val_loader  = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=test_batch_size, shuffle=False)

    os.makedirs("scoring_dir", exist_ok=True)
    # forward pass for dev
    # print("===> forward pass for dev set")
    # score_file_pth = os.path.join(data_files['scoring_dir'], str(pretrained_id) + '-epoch%s-dev_scores.txt' %(epoch_id))
    # print("===> dev scoring file saved at: '{}'".format(score_file_pth))
    score_file_pth="/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/score/LAD.txt"
    # prediction.prediction(val_loader, model, device, score_file_pth, data_files['dev_utt2systemID'])
    correct=0
    tot=0
    with torch.no_grad():
        # Iterate over the test dataset
        for i,(images,labels) in enumerate(val_loader):
            tot+=labels.size(0)
            # Forward pass through the model
            # images=images.to(dtype=torch.float32)
            batch_time = optimizer.AverageMeter()
            losses = optimizer.AverageMeter()
            top1 = optimizer.AverageMeter()
            total=0
            images=images.to(dtype=torch.float32)
            images=images.unsqueeze(1)
            images=images.to(device, non_blocking=True)
            labels=labels.to(device, non_blocking=True).view((-1,))
            print(labels)
            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)
            correct += (predicted_classes == labels).sum().item()
            print(outputs)
            f=open("outputs.txt","a")
            f.write(str(outputs))
            f.close()
            f=open("labels.txt","a")
            f.write(str(outputs))
            f.close()
            loss = F.nll_loss(outputs, labels)
            f=open("/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/score/loss.txt",'a')
            f.write(str(loss)+'\n')
            f.close()
            acc1, = optimizer.accuracy(outputs, labels, topk=(1, ))
            f=open("/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/score/acc.txt",'a')
            f.write(str(acc1)+'\n')
            f.close()
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))



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
            # correct += (predicted == labels).sum().item()
    print('===> Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    acc2=top1.avg
    accc=correct/tot
    f=open("/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/score/accc.txt",'a')
    f.write('\n'+str(accc))
    f.close()
    f=open("/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/score/acc2.txt",'a')
    f.write('\n'+str(acc2))
    f.close()
    # forward pass for eval
    # print("===> forward pass for eval set")
    # score_file_pth = os.path.join(data_files['scoring_dir'], str(pretrained_id) + '-epoch%s-eval_scores.txt' %(epoch_id))
    # print("===> eval scoring file saved at: '{}'".format(score_file_pth))
    # prediction.prediction(eval_loader, model, device, score_file_pth, data_files['eval_utt2systemID'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-feats', action='store', type=str, default='pa_spec')
    # parser.add_argument('--pretrained', action='store', type=str, default=None)
    parser.add_argument('--configfile', action='store', type=str)
    parser.add_argument('--random-seed', action='store', type=int, default=0)
    args = parser.parse_args()

    # pretrained = args.pretrained
    random_seed = args.random_seed

    with open(args.configfile) as json_file:
        config = json.load(json_file)

    data_files = datafiles.data_prepare[args.data_feats]
    model_params = config['model_params']
    training_params = config['training_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

    ''' 
    print(run_id)
    print(pretrained)
    print(data_files)
    print(model_params)
    print(training_params)
    print(device)
    exit(0)
    '''
    main(data_files, model_params, training_params, device)


