import torch
import os
import sys

sys.path.append('..')
import lib.tools as tools
from lib.normalization import imreadth, resize, normalize

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# pre-defined variables
use_cuda = torch.cuda.is_available()

half_precision = True  # use for memory saving

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='../trained_models/dualrc-net.pth.tar')
parser.add_argument('--aachen_path', type=str, default='../../storage/Aachen_Day_Night')
parser.add_argument('--image_size', type=int, default=1600)
parser.add_argument('--experiment_name', type=str, default='dualrc-net')
parser.add_argument('--skip_up_to', type=str, default='')
parser.add_argument('--nchunks', type=int, default=1)
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--Npts', type=int, default=8000)
parser.add_argument('--image_pairs', type=str, default='all')
parser.add_argument('--iter_step', type=int, default=1000)
parser.add_argument('--selection', type=str, default='partial')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--benchmark', type=bool, default=False)
args = parser.parse_args()
torch.cuda.set_device(args.device)
feature_extractor_device = args.device
print(args)

chp_args = torch.load(args.checkpoint)['args']
matcher = tools.ImgMatcher(use_cuda=use_cuda, half_precision=half_precision, checkpoint=args.checkpoint, postprocess_device=feature_extractor_device)

scale_factor = 0.0625

try:
    os.mkdir(os.path.join(args.aachen_path,'matches'))
except FileExistsError:
    pass

try:
    os.mkdir(os.path.join(args.aachen_path,'matches',args.experiment_name))
except FileExistsError:
    pass

# Get shortlists for each query image
if args.image_pairs=='all':
    pair_names_fn = os.path.join(args.aachen_path,'image_pairs_to_match.txt')
elif args.image_pairs=='queries':
    pair_names_fn = os.path.join(args.aachen_path,'query_pairs_to_match.txt')
elif args.image_pairs=='all_v1.1':
    pair_names_fn = os.path.join(args.aachen_path, 'image_pairs_to_match_v1_1.txt')
    
with open(pair_names_fn) as f:
    pair_names = [line.rstrip('\n') for line in f]

pair_names=np.array(pair_names)
pair_names_split = np.array_split(pair_names,args.nchunks)
pair_names_chunk = pair_names_split[args.chunk_idx]

pair_names_chunk=list(pair_names_chunk)

if args.skip_up_to!='':
    pair_names_chunk = pair_names_chunk[pair_names_chunk.index(args.skip_up_to)+1:]

for pair in tqdm(pair_names_chunk):
    src_fn = os.path.join(args.aachen_path,'database_and_query','images_upright',pair.split(' ')[0])
    src_image = plt.imread(src_fn)
    src=imreadth(src_fn)
    hA,wA=src.shape[-2:]
    src=resize(normalize(src), args.image_size, scale_factor)
    hA_,wA_=src.shape[-2:]

    tgt_fn = os.path.join(args.aachen_path,'database_and_query','images_upright',pair.split(' ')[1])
    tgt_image = plt.imread(tgt_fn)
    tgt=imreadth(tgt_fn)
    hB,wB=tgt.shape[-2:]
    tgt=resize(normalize(tgt), args.image_size, scale_factor)
    hB_,wB_=tgt.shape[-2:]

    with torch.no_grad():
        result, scores, features = matcher({'source_image':src, 'target_image':tgt}, num_pts=args.Npts, central_align=False, iter_step=args.iter_step, selection=args.selection, args=args)
        # pdb.set_trace()
        corr4d, featureA_0, featureB_0 = features
        fs1, fs2 = featureA_0.shape[2:]
        fs3, fs4 = featureB_0.shape[2:]
        ratio = int(np.round(hA_/fs1))
        xA_ = result[:, 0] / ratio
        yA_ = result[:, 1] / ratio
        xB_ = result[:, 2] / ratio
        yB_ = result[:, 3] / ratio
        score_ = scores

        YA,XA=torch.meshgrid(torch.arange(fs1),torch.arange(fs2))
        YB,XB=torch.meshgrid(torch.arange(fs3),torch.arange(fs4))

        YA = YA.contiguous()
        XA = XA.contiguous()
        YB = YB.contiguous()
        XB = XB.contiguous()

        YA=(YA+0.5)/(fs1)*hA
        XA=(XA+0.5)/(fs2)*wA
        YB=(YB+0.5)/(fs3)*hB
        XB=(XB+0.5)/(fs4)*wB

        XA = XA.view(-1).data.cpu().float().numpy()
        YA = YA.view(-1).data.cpu().float().numpy()
        XB = XB.view(-1).data.cpu().float().numpy()
        YB = YB.view(-1).data.cpu().float().numpy()

        keypoints_A=np.stack((XA,YA),axis=1)
        keypoints_B=np.stack((XB,YB),axis=1)

    #     idx_A = (yA_*fs2+xA_).long().view(-1,1)
    #     idx_B = (yB_*fs4+xB_).long().view(-1,1)
        idx_A = (yA_*fs2+xA_).view(-1,1)
        idx_B = (yB_*fs4+xB_).view(-1,1)
        score = score_.view(-1,1)

        matches = torch.cat((idx_A,idx_B,score),dim=1).cpu().numpy()

        kp_A_fn = src_fn+'.'+args.experiment_name
        kp_B_fn = tgt_fn+'.'+args.experiment_name

        if not os.path.exists(kp_A_fn):
            with open(kp_A_fn, 'wb') as output_file:
                np.savez(output_file,keypoints=keypoints_A)

        if not os.path.exists(kp_B_fn):
            with open(kp_B_fn, 'wb') as output_file:
                np.savez(output_file,keypoints=keypoints_B)

        matches_fn = pair.replace('/','-').replace(' ','--')+'.'+args.experiment_name
        matches_path = os.path.join(args.aachen_path,'matches',args.experiment_name,matches_fn)

        with open(matches_path, 'wb') as output_file:
            np.savez(output_file,matches=matches)
        print(matches_fn)

        del corr4d,src,tgt, featureA_0, featureB_0
        del xA_,xB_,yA_,yB_,score_
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
