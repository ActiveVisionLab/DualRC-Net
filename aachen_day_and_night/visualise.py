'''
This one is used to visualise the matching of aachen
'''
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2

import lib.tools as tools
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='../../storage/Aachen_Day_Night', help='Path to the dataset')
parser.add_argument('--method_name', type=str, default='dualrc-net', help='Name of the method')
args = parser.parse_args()

pair_file = 'query_pairs_to_match.txt'

pair_file = os.path.join(args.dataset_path, pair_file)
image_dir = os.path.join(args.dataset_path, 'database_and_query', 'images_upright')
match_dir = os.path.join(args.dataset_path, 'matches', args.method_name)

with open(pair_file, 'r') as f:
    image_pairs = f.readlines()

for pair in image_pairs[0:]:
    print(image_pairs.index(pair))
    img1, img2 = pair.strip().split(' ')
    img1_ = img1
    img2_ = img2

    # read in image
    img1 = os.path.join(image_dir, img1)
    img2 = os.path.join(image_dir, img2)
    img1 = plt.imread(img1)
    img2 = plt.imread(img2)

    # read in feature
    feature1 = os.path.join(image_dir, '%s.%s' % (img1_, args.method_name))
    feature2 = os.path.join(image_dir, '%s.%s' % (img2_, args.method_name))
    feature1 = np.load(feature1)['keypoints']
    feature2 = np.load(feature2)['keypoints']

    # read in matches
    name1 = img1_.replace('/', '-')
    name2 = img2_.replace('/', '-')
    matches = '%s--%s.%s' % (name1, name2, args.method_name)
    matches = np.load(os.path.join(match_dir, matches))['matches']
    matches = matches[:, :2].astype(np.uint32)

    pts1 = feature1[matches[:, 0]]
    pts2 = feature2[matches[:, 1]]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img2 = cv2.resize(img2, dsize=(int(w2*w1/w2), int(h2*w1/w2)))
    pts2 = pts2*w1/w2

    pts1 = pts1[:500, :]
    pts2 = pts2[:500, :]

    tools.draw_lines_for_matches(img1/255, img2/255, pts1, pts2)
    # plt.show()

    try:
        os.mkdir('/home/xinghui/storage/Aachen_{}/'.format(args.method_name))
    except FileExistsError:
        pass
    fn = '%s_vs_%s.pdf' % (name1, name2)
    fn2 = '%s_vs_%s.png' % (name1, name2)
    plt.savefig(os.path.join('/home/xinghui/storage/Aachen_{}/{}'.format(args.method_name, fn)),bbox_inches='tight')
    plt.savefig(os.path.join('/home/xinghui/storage/Aachen_{}/{}'.format(args.method_name, fn2)), bbox_inches='tight')
    plt.close()

