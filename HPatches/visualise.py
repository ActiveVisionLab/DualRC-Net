'''
Visualization of HPatches and store result in pdf file
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm import tqdm

import sys
sys.path.append('..')
import lib.tools

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../../storage/hpatches_sequences')
parser.add_argument('--method_name', type=str, default='dualrc-net')
args = parser.parse_args()

method_name = args.method_name
sequence_dir = os.path.join(args.root, 'hpatches-sequences-release')

def visualise(src, tgt, query, ref, result, t):
    good_query = query[result < t]
    good_ref = ref[result < t]
    bad_query = query[result >= t]
    bad_ref = ref[result >= t]

    print('correctly matches %i/%i' % (len(good_query), len(query)))

    # if src.shape[0] > src.shape[1]:
    #     fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 9))
    #     plt.subplots_adjust(hspace=0)
    # else:
    #     fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(9, 16))
    #     plt.subplots_adjust(wspace=0)
    #
    # axes[0].imshow(src)
    # # axes[0].set_title('original feature on reference image')
    # axes[0].scatter(good_query[:, 0], good_query[:, 1], s=2, c='lime')
    # axes[0].scatter(bad_query[:, 0], bad_query[:, 1], s=2, c='r')
    # axes[0].axis('off')
    # axes[1].imshow(tgt)
    # # axes[1].set_title('original feature on query image')
    # axes[1].scatter(good_ref[:, 0], good_ref[:, 1], s=4, c='lime')
    # axes[1].scatter(bad_ref[:, 0], bad_ref[:, 1], s=4, c='r')
    # axes[1].axis('off')

    gap = 20
    if src.shape[0] > src.shape[1]:
        fig = plt.figure(figsize=(21, 12))
        src = src / 255
        tgt = tgt / 255
        h1, w1, _ = src.shape
        h2, w2, _ = tgt.shape
        w = w1 + w2 + gap
        h = np.max((h1, h2))
        img = np.ones((h, w, 3))
        img[:h1, :w1, :] = src
        img[:h2, w1 + gap:w1 + gap + w2, :] = tgt

        good_ref[:, 0] = good_ref[:, 0] + gap + w1
        bad_ref[:, 0] = bad_ref[:, 0] + gap + w1

        plt.imshow(img)
        plt.scatter(good_query[:, 0], good_query[:, 1], s=2, c='lime')
        plt.scatter(bad_query[:, 0], bad_query[:, 1], s=2, c='r')
        plt.scatter(good_ref[:, 0], good_ref[:, 1], s=4, c='lime')
        plt.scatter(bad_ref[:, 0], bad_ref[:, 1], s=4, c='r')
    else:
        fig = plt.figure(figsize=(12, 21))
        src = src / 255
        tgt = tgt / 255
        h1, w1, _ = src.shape
        h2, w2, _ = tgt.shape
        h = h1 + h2 + gap
        w = np.max((w1, w2))
        img = np.ones((h, w, 3))
        img[:h1, :w1, :] = src
        img[h1 + gap:h1 + gap + h2, :w2, :] = tgt

        good_ref[:, 1] = good_ref[:, 1] + gap + h1
        bad_ref[:, 1] = bad_ref[:, 1] + gap + h1
        plt.imshow(img)
        plt.scatter(good_query[:, 0], good_query[:, 1], s=2, c='lime')
        plt.scatter(bad_query[:, 0], bad_query[:, 1], s=2, c='r')
        plt.scatter(good_ref[:, 0], good_ref[:, 1], s=4, c='lime')
        plt.scatter(bad_ref[:, 0], bad_ref[:, 1], s=4, c='r')
    plt.axis('off')
    return len(good_query), len(query)

match_dir = os.path.join(args.root, method_name)
matches_list = os.listdir(match_dir)
matches_list.sort()
for matches in tqdm(matches_list):
    sequence = matches.split('-')[0]
    src_id = matches.split('-')[1].split('.')[0].split('_')[0]
    tgt_id = matches.split('-')[1].split('.')[0].split('_')[1]

    # sequence = 'v_beyus'
    # src_id = '1'
    # tgt_id = '5'
    # matches = '{}-{}_{}.npz.{}'.format(sequence, src_id, tgt_id, method_name)

    src_image = plt.imread(os.path.join(sequence_dir, sequence,'{}.ppm'.format(src_id)))
    h1, w1 = src_image.shape[:2]
    tgt_image = plt.imread(os.path.join(sequence_dir, sequence, '{}.ppm'.format(tgt_id)))
    h2, w2 = src_image.shape[:2]

    H_file = os.path.join(sequence_dir, sequence, 'H_{}_{}'.format(src_id, tgt_id))
    H = np.loadtxt(H_file)

    matches_file = os.path.join(match_dir, matches)
    query = np.load(matches_file)['keypoints_A']
    ref = np.load(matches_file)['keypoints_B']
    score = np.load(matches_file)['scores']

    indice = np.argsort(-score)[:2000]
    query = query[indice]
    ref = ref[indice]

    npts = query.shape[0]
    query_ = np.concatenate((query, np.ones((npts, 1))), axis=1)
    projection = np.matmul(H, query_.T).T
    projection = projection / projection[:, 2:3]
    projection = projection[:, :2]
    result = np.linalg.norm(ref - projection, axis=1)

    h1_ = int(1600/max(h1, w1) * h1)
    w1_ = int(1600 / max(h1, w1) * w1)
    h2_ = int(1600/max(h2, w2) * h2)
    w2_ = int(1600 / max(h2, w2) * w2)
    src_image = cv2.resize(src_image, dsize=(w1_, h1_))
    tgt_image = cv2.resize(tgt_image, dsize=(w2_, h2_))
    query = query * (h1_ / h1)
    ref = ref * (h2_ / h2)

    num_correct, num_total = visualise(src_image, tgt_image, query, ref, result, 3)
    save_dir = os.path.join(args.root, 'hpatches_{}_2000'.format(method_name))
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    save_path = os.path.join(save_dir, '{}_{}_{}_({}-{}).pdf'.format(sequence, src_id, tgt_id, num_correct, num_total))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


