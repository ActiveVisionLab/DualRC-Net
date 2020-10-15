import torch
import os
import sys
sys.path.append('..')

from lib.normalization import imreadth, resize, normalize
import lib.tools as tools

import numpy as np
import numpy.random
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
import argparse

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='../trained_models/dualrc-net.pth.tar')
parser.add_argument('--inloc_shortlist', type=str, default='densePE_top100_shortlist_cvpr18.mat')
parser.add_argument('--pano_path', type=str, default='../../storage/InLoc/cutouts/', help='path to InLoc panos - should contain CSE3,CSE4,CSE5,DUC1 and DUC2 folders')
parser.add_argument('--query_path', type=str, default='../../storage/InLoc/iphone7/', help='path to InLoc queries')
parser.add_argument('--image_size', type=int, default=1600)
parser.add_argument('--experiment_name', type=str, default='dualrc-net')
parser.add_argument('--nchunks', type=int, default=1)
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--skip_up_to', type=str, default='')
parser.add_argument('--relocalize', type=int, default=1)
parser.add_argument('--Npts', type=int, default=2000)
parser.add_argument('--n_queries', type=int, default=356)
parser.add_argument('--n_panos', type=int, default=10)
parser.add_argument('--iter_step', type=int, default=1000)
parser.add_argument('--selection', type=str, default='partial')
parser.add_argument('--im_fe_ratio', type=int, default=16)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--benchmark', type=bool, default=False)
args = parser.parse_args()

torch.cuda.set_device(args.device)
feature_extractor_device = args.device
print(args)

im_fe_ratio = 16
half_precision = True

model = tools.ImgMatcher(use_cuda=use_cuda, half_precision=half_precision, checkpoint=args.checkpoint, postprocess_device=feature_extractor_device, im_fe_ratio=args.im_fe_ratio)

# Generate output folder path
output_folder = args.experiment_name + '_'+ args.selection
print('Output matches folder: '+output_folder)

scale_factor = 0.0625

# Get shortlists for each query image
shortlist_fn = args.inloc_shortlist

dbmat = loadmat(shortlist_fn)
db = dbmat['ImgList'][0,:]

query_fn_all=np.squeeze(np.vstack(tuple([db[q][0] for q in range(len(db))])))
pano_fn_all=np.vstack(tuple([db[q][1] for q in range(len(db))]))

Nqueries=args.n_queries
Npanos=args.n_panos

try:
    os.mkdir(os.path.join(os.getcwd(), 'matches'))
except FileExistsError:
    pass

try:
    os.mkdir(os.path.join(os.getcwd(), 'matches', output_folder))
except FileExistsError:
    pass

queries_idx = np.arange(Nqueries)
queries_idx_split = np.array_split(queries_idx,args.nchunks)
queries_idx_chunk = queries_idx_split[args.chunk_idx]

queries_idx_chunk=list(queries_idx_chunk)
if args.skip_up_to!='':
    skip_up_to = int(args.skip_up_to)
    queries_idx_chunk = queries_idx_chunk[queries_idx_chunk.index(skip_up_to)+1:]

indices = range(Npanos)
for q in queries_idx_chunk:
    print(q)
    matches=numpy.zeros((1,Npanos,args.Npts,5))
    # load query image
    src_fn = os.path.join(args.query_path,db[q][0].item())
    src=imreadth(src_fn)
    hA,wA=src.shape[-2:]
    src=resize(normalize(src), args.image_size, scale_factor)
    hA_,wA_=src.shape[-2:]

    with torch.no_grad():
    # load database image
        for idx in indices:

            tgt_fn = os.path.join(args.pano_path,db[q][1].ravel()[idx].item())
            tgt=imreadth(tgt_fn)
            hB,wB=tgt.shape[-2:]
            tgt=resize(normalize(tgt), args.image_size, scale_factor)
            hB_,wB_=tgt.shape[-2:]
            result, scores, _ = model({'source_image': src, 'target_image': tgt}, num_pts=args.Npts, central_align=True, iter_step=args.iter_step, selection=args.selection, args=args)
            H_src, W_src = src.shape[-2:]
            H_tgt, W_tgt = tgt.shape[-2:]

            # if the keypoint obtained is less than the required number, fill the gap with the last element
            if result.shape[0] < args.Npts:
                num = result.shape[0]
                diff = args.Npts - num
                fill_result = result[-1].expand(diff, -1)
                fill_scores = scores[-1].expand(diff)
                result = torch.cat((result, fill_result), dim=0)
                scores = torch.cat((scores, fill_scores), dim=0)

            xA_ = result[:, 0] / W_src
            yA_ = result[:, 1] / H_src
            xB_ = result[:, 2] / W_tgt
            yB_ = result[:, 3] / H_tgt

            xA = xA_.view(-1).data.cpu().float().numpy()
            yA = yA_.view(-1).data.cpu().float().numpy()
            xB = xB_.view(-1).data.cpu().float().numpy()
            yB = yB_.view(-1).data.cpu().float().numpy()
            score = scores.view(-1).data.cpu().float().numpy()

            matches[0,idx,:,0]=xA
            matches[0,idx,:,1]=yA
            matches[0,idx,:,2]=xB
            matches[0,idx,:,3]=yB
            matches[0,idx,:,4]=score

            # uncomment the following part if want to draw the first 500 matches
            # s = W_src
            # result = result[:500]
            # tgt = nn.functional.interpolate(tgt, (int(H_tgt /  W_tgt * s), int(W_tgt / W_tgt * s)), mode='bilinear', align_corners=True)
            # src_image = tools.pre_draw(src)
            # tgt_image = tools.pre_draw(tgt)
            # tools.draw_lines_for_matches(tgt_image, src_image, result[:, 2:] / np.max((H_tgt, W_tgt)) * s, result[:, :2])
            # try:
            #     os.mkdir('/home/xinghui/storage/inloc_%s_%i/' % (args.experiment_name, 500))
            # except FileExistsError:
            #     pass
            # fn = '%s_vs_%s.png' % (db[q][0].item(), db[q][1].ravel()[idx].item().replace('/', '-'))
            # fn2 = '%s_vs_%s.pdf' % (db[q][0].item(), db[q][1].ravel()[idx].item().replace('/', '-'))
            # plt.savefig(os.path.join('/home/xinghui/storage/inloc_%s_%i/' % (args.experiment_name, 500) , fn),bbox_inches='tight')
            # plt.savefig(os.path.join('/home/xinghui/storage/inloc_%s_%i/' % (args.experiment_name, 500) , fn2),bbox_inches='tight')
            # plt.close()

            del tgt
            del xA,xB,yA,yB,score
            del xA_,xB_,yA_,yB_,scores
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

            print(">>>"+str(idx))

        matches_file=os.path.join('../datasets/inloc/matches/',output_folder,str(q+1)+'.mat')
        savemat(matches_file,{'matches':matches,'query_fn':db[q][0].item(),'pano_fn':pano_fn_all},do_compression=True)
        print(matches_file)
        del src
