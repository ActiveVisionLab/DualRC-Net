'''
this script is to plot our result with sparse ncnet in Hpatches dataset
'''
import numpy as np
import matplotlib.pyplot as plt

# create a dictionary that contain all result
files = {
    'DualRC-Net' : 'cache-top/dualrc-net_1600_1000.txt',
    'Sparse-NCNet (1600 H+S)' : 'cache-top/ncnet.sparsencnet_1600_hard_soft_2k.npy',
    'NCNet (3200 H)' : 'cache-top/ncnet.densencnet_1600_hard_2k.npy',
    'D2-Net' : 'cache-top/d2-net.npy',
    'DELF' : 'cache-top/delf.npy',
    'R2D2' : 'cache-top/r2d2.npy',
    'SP': 'cache-top/superpoint.npy',
    'SP+SG': 'cache-top/sp_sg_ImgSize1600_MaxNum2000_SGThres0_NMS2.txt',
}

marker = ['D--r',
          'x-k',
          '<-C7',
          '|-C0',
          'h-C1',
          '>-C2',
          '+-C3',
          'o-C4']

def read_npy(file):
    data = np.load(file, allow_pickle=True)
    type, num = np.unique(data[2][0], return_counts=True)
    count_dict = dict(zip(type, num))
    num_i = count_dict['i']
    num_v = count_dict['v']

    MMA_i_ = np.array(list(data[0].values())[:10])
    MMA_v_ = np.array(list(data[1].values())[:10])
    MMA_i = MMA_i_ / num_i
    MMA_v = MMA_v_ / num_v
    MMA = (MMA_i_ + MMA_v_) / (num_i + num_v)

    return MMA_i, MMA_v, MMA

def read_txt(our_file):
    with open(our_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] == 'MMA_i':
                MMA_i = np.array(line[1:], dtype=float)
            if line[0] == 'MMA_v':
                MMA_v = np.array(line[1:], dtype=float)
            if line[0] == 'MMA':
                MMA = np.array(line[1:], dtype=float)
    return MMA_i, MMA_v, MMA

x = np.linspace(1, 10, num=10 ,endpoint=True)

linewidth = 8
markersize = 15
title_size = 35
label_size = 35
tick_size = 20
position = 0.15

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(27, 7.5))

for i, (key, path) in enumerate(files.items()):
    if path[-3:] == 'txt':
        MMA_i, MMA_v, MMA = read_txt(path)
    elif path[-3:] == 'npy':
        MMA_i, MMA_v, MMA = read_npy(path)
    else:
        assert 'File format error'

    axes[0].plot(x, MMA_i, marker[i], label=key, linewidth=linewidth, markersize=markersize)
    axes[1].plot(x, MMA_v, marker[i], label=key, linewidth=linewidth, markersize=markersize)
    axes[2].plot(x, MMA, marker[i], label=key, linewidth=linewidth, markersize=markersize)

axes[0].set_title('Illumination', size=title_size)
axes[0].set_xlabel('threshold [px]', fontsize=label_size)
axes[0].set_ylabel('MMA', fontsize=label_size)
axes[0].set_xlim([1, 10])
axes[0].set_ylim([0, 1])
axes[0].grid()
axes[0].tick_params(axis='x', labelsize=tick_size)
axes[0].tick_params(axis='y', labelsize=tick_size)
pos1 = axes[0].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 + position,  pos1.width, pos1.height]
axes[0].set_position(pos2) # set a new position

axes[1].set_title('Viewpoint', size=title_size)
axes[1].set_xlabel('threshold [px]', fontsize=label_size)
axes[1].set_ylabel('MMA', fontsize=label_size)
axes[1].set_xlim([1, 10])
axes[1].set_ylim([0, 1])
axes[1].grid()
axes[1].tick_params(axis='x', labelsize=tick_size)
axes[1].tick_params(axis='y', labelsize=tick_size)
pos1 = axes[1].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 + position,  pos1.width, pos1.height]
axes[1].set_position(pos2) # set a new position

axes[2].set_title('Overall', size=title_size)
axes[2].set_xlabel('threshold [px]', fontsize=label_size)
axes[2].set_ylabel('MMA', fontsize=label_size)
axes[2].set_xlim([1, 10])
axes[2].set_ylim([0, 1])
axes[2].grid()
axes[2].tick_params(axis='x', labelsize=tick_size)
axes[2].tick_params(axis='y', labelsize=tick_size)
pos1 = axes[2].get_position() # get the original position
pos2 = [pos1.x0, pos1.y0 + position,  pos1.width, pos1.height]
axes[2].set_position(pos2) # set a new position

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=8, labelspacing=0.7, fontsize=35, frameon=False, columnspacing=1, borderaxespad=0)
plt.show()

