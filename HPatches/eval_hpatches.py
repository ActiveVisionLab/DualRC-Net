import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import pdb
import time
import cv2

sys.path.append('..')
import lib.tools as tools
from lib.normalization import imreadth, resize, normalize


use_cuda = torch.cuda.is_available()

im_fe_ratio = 16
half_precision = True

torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

Transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser(description='Evaluating HPatches Dataset')
parser.add_argument('--checkpoint', type=str, default='../trained_models/dualrc-net.pth.tar')
parser.add_argument('--experiment_name', type=str, default='dualrc-net')
parser.add_argument('--root', type=str, default='../../storage/', help='to the parent folder of hpatches_sequences folder')
parser.add_argument('--sequence_list', type=str, default='image_list_hpatches_sequences.txt')
parser.add_argument('--image_size', type=int, default=1600, help='image size used')
parser.add_argument('--Npts', type=int, default=2000, help='how many matches selected')
parser.add_argument('--iter_step', type=int, default=1000)
parser.add_argument('--selection', type=str, default='partial')
parser.add_argument('--im_fe_ratio', type=int, default=16)
parser.add_argument('--device', type=int, default=0, help='which gpu should the experiment be run on')
parser.add_argument('--benchmark', type=bool, default=False, help='whether to benchmark the speed. If does, it will use the first image for 20 times')
args = parser.parse_args()

torch.cuda.set_device(args.device)
feature_extractor_device = args.device

matcher = tools.ImgMatcher(use_cuda=use_cuda, half_precision=half_precision, checkpoint=args.checkpoint, postprocess_device=feature_extractor_device, im_fe_ratio=args.im_fe_ratio)

if not args.benchmark:
    experiment_name = '%s_%d_%d_%s' % (args.experiment_name, args.image_size, args.Npts, args.selection)
    output_name = 'hpatches_%s_%d_%d_%s.txt' % (args.experiment_name, args.image_size, args.Npts, args.selection)
    out = open(output_name, 'w')
    print(output_name)

im_list_ = open(args.sequence_list, 'r')
im_list = im_list_.readlines()

MMA_v = np.zeros(10)
MMA_i = np.zeros(10)
MMA = np.zeros(10)
num_v = 0
num_i = 0

scale_factor = 0.0625
running_time = 0
counter = 0

with torch.no_grad():
    for i in tqdm(range(0*6, len(im_list), 6)):
        if args.benchmark:
            i = 0
        scene_list = im_list[i: i+6]
        scene_list.sort()
        change_type = scene_list[0].strip().split('/')[-2].split('_')[0]
        scene = scene_list[0].strip().split('/')[-2]
        for j in range(1, len(scene_list)):     # pair the first image with the rest of 5 images
            if args.benchmark:
                j = 1
            ref_im = scene_list[j].strip()
            query_im = scene_list[0].strip()

            scene_dir = ref_im.split('/')
            scene_dir = '/'.join(scene_dir[:-1])

            # completely define various path
            query_im = os.path.join(args.root, query_im)
            query_im_ = query_im
            ref_im = os.path.join(args.root, ref_im)
            scene_dir = os.path.join(args.root, scene_dir)
            H_file = os.path.join(args.root, scene_dir, 'H_1_%s'%str(j+1))

            query_im = imreadth(query_im)
            hA, wA = query_im.shape[-2:]
            query_im = resize(normalize(query_im), args.image_size, scale_factor)
            hA_, wA_ = query_im.shape[-2:]

            ref_im = imreadth(ref_im)
            hB, wB = ref_im.shape[-2:]
            ref_im = resize(normalize(ref_im), args.image_size, scale_factor)
            hB_, wB_ = ref_im.shape[-2:]

            # create batch
            batch ={}
            batch['source_image'] = query_im.cuda()
            batch['target_image'] = ref_im.cuda()

            start = time.time()
            matches, score, _ = matcher(batch, num_pts=args.Npts, central_align=True, iter_step=args.iter_step, selection=args.selection)
            end = time.time()

            if args.benchmark:
                print('total time %f second' % (end - start))
                counter += 1
                running_time += end-start
                print('memory allocated ', torch.cuda.max_memory_allocated()/1024/1024)
                if counter == 20:
                    print('Average time %f second' % (running_time / counter))
                    exit()

            matches = matches.cpu().numpy()
            score = score.view(-1).cpu().numpy()

            # read in homography
            H = np.loadtxt(H_file)
            # project the query to reference
            npts = matches.shape[0]
            query = matches[:, :2] * (hA / hA_)
            ref = matches[:, 2:] * (hB / hB_)
            query_ = np.concatenate((query, np.ones((npts, 1))), axis=1)
            projection = np.matmul(H, query_.T).T

            # convert the projection from homogeneous coordinate to inhomogeneous coordinate
            projection = projection / projection[:, 2:3]
            projection = projection[:, :2]
            # evaluate the result
            result = np.linalg.norm(ref-projection, axis=1)

            if not args.benchmark:
                # save matches
                save_dir = os.path.join(args.root, 'hpatches_sequences', experiment_name)
                try:
                    os.mkdir(save_dir)
                except FileExistsError:
                    pass

                matches_file = '{}-{}_{}.npz.{}'.format(scene, '1', str(j+1), experiment_name)

                with open(os.path.join(save_dir,matches_file), 'wb') as output_file:
                    np.savez(
                            output_file,
                            keypoints_A=query,
                            keypoints_B=ref,
                            scores=score
                    )
                print(matches_file)

            for thres in range(1, 11):
                idx = thres-1
                if change_type == 'v':
                    MMA_v[idx] += np.sum(result <= thres)/result.shape[0]
                if change_type == 'i':
                    MMA_i[idx] += np.sum(result <= thres) / result.shape[0]
                MMA[idx] += np.sum(result <= thres) / result.shape[0]


            if change_type == 'v':
                num_v += 1
            if change_type == 'i':
                num_i += 1

            del matches, projection
            del batch, query_im, ref_im

            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

        # exit()
        print(MMA_i)
        print(MMA_v)
        print(MMA)
    MMA_i = MMA_i / num_i
    MMA_v = MMA_v / num_v
    MMA = MMA / (num_i + num_v)
    print('MMA_i', MMA_i)
    print('MMA_v', MMA_v)
    print('MMA', MMA)
    MMA_i = MMA_i.tolist()
    MMA_v = MMA_v.tolist()
    MMA = MMA.tolist()
    MMA_i = [str(i) for i in MMA_i]
    MMA_v = [str(i) for i in MMA_v]
    MMA = [str(i) for i in MMA]

    if not args.benchmark:
        out.write('MMA_i %s\n' % ' '.join(MMA_i))
        out.write('MMA_v %s\n' % ' '.join(MMA_v))
        out.write('MMA %s' % ' '.join(MMA))

    out.close()
    im_list_.close()