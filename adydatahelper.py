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
train_data = dataaa(typee='LA',e='eval')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
s=0
b=0
for i,(images,labels) in enumerate(train_loader):
	# if labels==1:
	# 	s=s+1
	# elif labels==0:
	# 	b=b+1
	# print(labels)
	# print(len(i))
	images=images.to(dtype=torch.float32)
	images=images.unsqueeze(1)
	print(images.shape)
# print(b,s)