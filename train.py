import os
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from lib.dataset_strong_loss import StrongLossDataset
from tqdm import tqdm
from lib.loss import SparseStrongWeakLoss

from torch.utils.data import DataLoader
from lib.model_v2 import ImMatchNet
from torch.utils.tensorboard import SummaryWriter

import argparse

# set visible gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

# Argument parsing
parser = argparse.ArgumentParser(description='Dual-Resolution Correspondence Network Training script')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training_file', type=str, default='../storage/MegaDepth_v1_SfM/training_pairs.txt')
parser.add_argument('--validation_file', type=str, default='../storage/MegaDepth_v1_SfM/validation_pairs.txt')
parser.add_argument('--image_path', type=str, default='../storage/MegaDepth_v1_SfM')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--num_epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[3, 3], help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16, 1], help='channels in neigh. cons')
parser.add_argument('--result_model_fn', type=str, default='dualrcnet', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params',  type=int, default=0, help='number of layers to finetune')
parser.add_argument('--multi_gpu', type=bool, default=True, help='whether to use multi gpu to train the model')

parser.add_argument('--use_scheduler', type=bool, default=True, help='whether to use optmizer lr scheduler')
parser.add_argument('--scheduler_milestone', type=int, nargs='+', default = [5, 10, 15])
parser.add_argument('--use_writer', type=bool, default=False, help='whether to use tensorboard to record the loss')

parser.add_argument('--backbone', type=str, default='resnet101fpn_3_1024_4', help='backbone used to extract feature map')
parser.add_argument('--im_fe_ratio', type=int, default=16, help='The ratio between image resolution and coarse level femap')
parser.add_argument('--fine_coarse_ratio', type=int, default=4, help='The ratio between fine level femap and coarse level femap')

parser.add_argument('--weight_loss', nargs='+', type=float, default = [0., 1, 0 ],help='the weight for sparse strong and weak loss ' )
parser.add_argument('--mode', type=int, default=1, help='0: 1NN, 1:4NN')
parser.add_argument('--loss', type=str, default='orthogonal_meanfnorm',help='specify the type of loss: meanfnorm, orthogonal_meanfnorm, displacement, balanced')
parser.add_argument('--gauss_size', type=int, default = 3, help='blur ground truth, it can be 0 3 5 7')
parser.add_argument('--weight_orthogonal', type=float, default=0.05,help='the weight for orthogonal term.')
parser.add_argument('--numKey', type=int, default=128, help='number of keypoints for each image pair')

args = parser.parse_args()
print(args)

im_fe_ratio = args.im_fe_ratio
model = ImMatchNet(use_cuda=use_cuda, multi_gpu=args.multi_gpu, ncons_kernel_sizes=args.ncons_kernel_sizes,
                   ncons_channels=args.ncons_channels, checkpoint=args.checkpoint, feature_extraction_cnn=args.backbone)
if args.multi_gpu:
    model = model.cuda()
    model = nn.DataParallel(model)


# Set which parts of the model to train
if args.fe_finetune_params>0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i+1)].parameters():
            p.requires_grad=True

print('Trainable parameters:')
for i,p in enumerate(filter(lambda p: p.requires_grad, model.parameters())):
    print(str(i+1)+": "+str(p.shape))

print('Using Adam optimizer')
optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
if args.use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestone, gamma=0.5)

# Define checkpoint_name
checkpoint_name = os.path.join(args.result_model_dir, datetime.datetime.now().strftime(
                               "%Y-%m-%d_%H_%M") + '_' + args.result_model_fn + '_%s_gauKer=%d_mode=%d' %
                               (args.backbone,args.gauss_size, args.mode) +'.pth.tar')

# build transform
transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_set = StrongLossDataset(file=args.training_file, image_path=args.image_path, transforms=transformer)
validation_set = StrongLossDataset(file=args.validation_file, image_path=args.image_path, transforms=transformer)

# build dataloader
training_loader = DataLoader(training_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=4, shuffle=True)

if args.multi_gpu:
    model.module.FeatureExtraction.eval()
else:
    model.FeatureExtraction.eval()

# create Tensorboard writer
if args.use_writer:
    writer = SummaryWriter('logs/MegaDepth/' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")+'_'+args.result_model_fn)

# create strongly supervised loss
loss_fn = SparseStrongWeakLoss ( image_size = args.image_size, model = model, loss_name = args.loss, backbone=args.backbone,
                     weight_orthogonal=args.weight_orthogonal, weight_loss = args.weight_loss, fine_coarse_ratio=args.fine_coarse_ratio,
                     im_fe_ratio = im_fe_ratio, gauss_size = args.gauss_size, mode = args.mode, N=args.numKey)

best = float("inf")
PCK_best = 0
for epoch in tqdm(range(args.num_epochs)):
    epoch = epoch+1
    running_loss = 0

    for idx, batch in tqdm(enumerate(training_loader)):
        batch['source_image'] = batch['source_image'].cuda()
        batch['target_image'] = batch['target_image'].cuda()
        batch['source_points'] = batch['source_points'].cuda()
        batch['target_points'] = batch['target_points'].cuda()
        batch['assignment'] = batch['assignment'].cuda()
        optimizer.zero_grad()
        loss, _, _, _, _ = loss_fn(batch)

        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        print('epoch', epoch, 'batch', idx, 'batch training loss', loss_item, 'lr', optimizer.param_groups[0]['lr'])
        running_loss += loss_item
        if args.use_writer:
            writer.add_scalar('training_loss', loss_item, (epoch-1) * len(training_loader) + idx)

    train_mean_loss = running_loss / len(training_loader)

    with torch.no_grad():
        running_PCK = 0
        running_loss = 0
        # model.eval()
        for idx, batch in tqdm(enumerate(validation_loader)):
            batch['source_image'] = batch['source_image'].cuda()
            batch['target_image'] = batch['target_image'].cuda()
            batch['source_points'] = batch['source_points'].cuda()
            batch['target_points'] = batch['target_points'].cuda()
            batch['assignment'] = batch['assignment'].cuda()

            loss, _, _, _, _ = loss_fn(batch)

            loss_item = loss.item()
            running_loss += loss_item

        val_mean_loss = running_loss / len(validation_loader)

    is_best = val_mean_loss < best

    if is_best:
        best = val_mean_loss

    print('validation_loss', val_mean_loss)
    if args.use_writer:
        writer.add_scalar('validation_loss', val_mean_loss, epoch-1)
    if args.use_scheduler:
        scheduler.step()
    dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'args': args,
        'optimizer': optimizer.state_dict(),
        'training_loss': train_mean_loss,
        'validation_loss': val_mean_loss
    }

    dirname = os.path.dirname(checkpoint_name)
    basename = os.path.basename(checkpoint_name)

    print('is best?', is_best)
    if is_best:
        print('saving best model...')
        torch.save(dict, os.path.join(dirname, 'best_'+basename))
