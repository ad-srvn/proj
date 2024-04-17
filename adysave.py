# from local import datafiles, trainer, validate, optimizer
import torch
checkpoint=torch.load("/Volumes/Seagate/ASV-anti-spoofing-with-Res2Net/model_snapshots/SEResNet34Debugfeats0/model_best.pth.tar")
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(checkpoint["best_acc1"])