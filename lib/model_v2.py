from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lib.conv4d import Conv4d

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='resnet101', feature_extraction_model_file='',
                 normalization=True, last_layer='', use_cuda=True, device = 0, tune_fusing_layer = False):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn = feature_extraction_cnn
        self.device = device

        # for resnet
        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        if feature_extraction_cnn == 'resnet101fpn_3_1024_4':   # feature with 1024 feature
            # left only one 3x3 conv after the final merge
            model = models.resnet101(pretrained=True)
            resnet_module_list = [getattr(model, l) for l in resnet_feature_layers]
            out_channels = 1024
            selected_layer_list = ['layer1', 'layer2', 'layer3']
            in_channels_list = [256, 512, 1024]
            in_channels_list.reverse()  # reverse the direction so can be used up to bottom
            selected_layer_idx_list = []
            self.extractor_list = nn.ModuleList()
            self.last_layer_block = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)    # 3x3 conv for the fine resolution feature map
            self.inner_block_list = nn.ModuleList()
            for layer in selected_layer_list:
                index = resnet_feature_layers.index(layer)
                selected_layer_idx_list.append(index)

            # build backbone
            self.extractor_list.append(nn.Sequential(*resnet_module_list[:selected_layer_idx_list[0]+1]))
            for i in range(1, len(selected_layer_idx_list)):
                current_index = selected_layer_idx_list[i]
                last_index = selected_layer_idx_list[i-1]
                self.extractor_list.append(nn.Sequential(*resnet_module_list[last_index+1: current_index+1]))
            for extractor in self.extractor_list:   # Fix backbone
                for param in extractor.parameters():
                    param.requires_grad = False

            # create inner block
            for in_channels in in_channels_list[1:]:
                self.inner_block_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

            if use_cuda:
                for i in range(len(self.extractor_list)):
                    self.extractor_list[i] = self.extractor_list[i].cuda()
                for i in range(len(self.inner_block_list)):
                    self.inner_block_list[i] = self.inner_block_list[i].cuda()
                self.last_layer_block = self.last_layer_block.cuda()

        if feature_extraction_cnn == 'resnet101fpn_3_256_4':    # feature with 256 channels
            # left only one 3x3 conv after the final merge
            model = models.resnet101(pretrained=True)
            resnet_module_list = [getattr(model, l) for l in resnet_feature_layers]
            out_channels = 256
            selected_layer_list = ['layer1', 'layer2', 'layer3']
            in_channels_list = [256, 512, 1024]
            in_channels_list.reverse()  # reverse the direction so can be used up to bottom

            selected_layer_idx_list = []
            self.extractor_list = nn.ModuleList()
            self.coarse_layer_block = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)      # 3x3 conv for the coarse resolution feature map
            self.fine_layer_block = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)      # 3x3 conv for the fine resolution feature map
            self.inner_block_list = nn.ModuleList()
            self.coarse_layer_idx = 0  # layer4 in reverse direction
            self.fine_layer_idx = 2  # layer1 in reverse direction
            for layer in selected_layer_list:
                index = resnet_feature_layers.index(layer)
                selected_layer_idx_list.append(index)

            # build backbone
            self.extractor_list.append(nn.Sequential(*resnet_module_list[:selected_layer_idx_list[0]+1]))
            for i in range(1, len(selected_layer_idx_list)):
                current_index = selected_layer_idx_list[i]
                last_index = selected_layer_idx_list[i-1]
                self.extractor_list.append(nn.Sequential(*resnet_module_list[last_index+1: current_index+1]))
            for extractor in self.extractor_list:   # Fix backbone
                for param in extractor.parameters():
                    param.requires_grad = False

            # create inner block
            for in_channels in in_channels_list:
                self.inner_block_list.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

            if use_cuda:
                for i in range(len(self.extractor_list)):
                    self.extractor_list[i] = self.extractor_list[i].cuda()
                for i in range(len(self.inner_block_list)):
                    self.inner_block_list[i] = self.inner_block_list[i].cuda()
                self.coarse_layer_block = self.coarse_layer_block.cuda()
                self.fine_layer_block = self.fine_layer_block.cuda()

        if train_fe == False and not 'fpn' in feature_extraction_cnn :
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

        if use_cuda and not 'fpn' in feature_extraction_cnn:
            self.model = self.model.cuda()

    def forward(self, image_batch):

        if self.feature_extraction_cnn == 'resnet101fpn_3_1024_4':
            bottom_up = []
            for extractor in self.extractor_list:
                image_batch = extractor(image_batch)
                bottom_up.append(image_batch)

            up_bottom = [image_batch]
            # last_layer = image_batch
            bottom_up.reverse()     # reverse to go from up to bottom

            for i in range(1, len(bottom_up)):
                inner_block = self.inner_block_list[i-1]
                last_layer = up_bottom[-1]
                current_layer = bottom_up[i]
                current_shape = current_layer.shape[-2:]
                current_layer = inner_block(current_layer) + F.interpolate(last_layer, size=current_shape, mode='bilinear', align_corners=True)
                if i == len(bottom_up) - 1:
                    current_layer = self.last_layer_block(current_layer)
                up_bottom.append(current_layer)
            if self.feature_extraction_cnn == 'resnet101fpn_3_1024_4':
                return up_bottom[-1], up_bottom[0]
            else:
                return up_bottom[-1], up_bottom[0], self.pooling(up_bottom[-1])

        elif self.feature_extraction_cnn == 'resnet101fpn_3_256_4':
            bottom_up = []
            for extractor in self.extractor_list:
                image_batch = extractor(image_batch)
                bottom_up.append(image_batch)

            top_inner_block = self.inner_block_list[0]
            up_bottom = [top_inner_block(image_batch)]
            # last_layer = image_batch
            bottom_up.reverse()     # reverse to go from up to bottom

            for i in range(1, len(bottom_up)):
                inner_block = self.inner_block_list[i]
                last_layer = up_bottom[-1]
                current_layer = bottom_up[i]
                current_shape = current_layer.shape[-2:]
                current_layer = inner_block(current_layer) + F.interpolate(last_layer, size=current_shape, mode='bilinear', align_corners=True)
                up_bottom.append(current_layer)

            fine_output = self.fine_layer_block(up_bottom[self.fine_layer_idx])
            coarse_output = self.coarse_layer_block(up_bottom[self.coarse_layer_idx])
            return fine_output, coarse_output


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True, device=0):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()
        self.device = device

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B, feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True, device = 0):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.device = device
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output

    # del corr4d_A, corr4d_B, corr4d_A_max, corr4d_B_max
    return corr4d


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:, 0, i::k_size, j::k_size, k::k_size, l::k_size].unsqueeze(1))
    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


class ImMatchNet(nn.Module):
    def __init__(self,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 feature_extraction_model_file=None,
                 return_correlation=False,
                 ncons_kernel_sizes=[3, 3, 3],
                 ncons_channels=[10, 10, 1],
                 normalize_features=True,
                 train_fe=False,
                 use_cuda=True,
                 multi_gpu=False,
                 half_precision=False,
                 checkpoint=None,
                 device=0,
                 ):

        super(ImMatchNet, self).__init__()
        # Load checkpoint
        if checkpoint is not None and checkpoint is not '':
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict(
                [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])

            # override relevant parameters
            print('Using checkpoint parameters: ')
            if hasattr(checkpoint['args'], 'backbone'):
                feature_extraction_cnn = checkpoint['args'].backbone
            ncons_channels = checkpoint['args'].ncons_channels
            print('  ncons_channels: ' + str(ncons_channels))
            ncons_kernel_sizes = checkpoint['args'].ncons_kernel_sizes
            print('  ncons_kernel_sizes: ' + str(ncons_kernel_sizes))

        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.half_precision = half_precision
        self.feature_extraction_cnn = feature_extraction_cnn
        self.device = device

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_model_file=feature_extraction_model_file,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)

        self.FeatureCorrelation = FeatureCorrelation(shape='4D', normalization=False)


        print('Using NC Module')
        self.NeighConsensus = NeighConsensus(use_cuda=self.use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels)

        # Load weights
        if checkpoint is not None and checkpoint is not '':
            print('Copying weights...')
            for name, param in self.FeatureExtraction.state_dict().items():
                if 'num_batches_tracked' not in name:
                    if multi_gpu:
                        self.FeatureExtraction.state_dict()[name].copy_(
                            checkpoint['state_dict']['module.FeatureExtraction.' + name])
                    else:
                        self.FeatureExtraction.state_dict()[name].copy_(
                            checkpoint['state_dict']['FeatureExtraction.' + name])
            for name, param in self.NeighConsensus.state_dict().items():
                if multi_gpu:
                    self.NeighConsensus.state_dict()[name].copy_(
                        checkpoint['state_dict']['module.NeighConsensus.' + name])
                else:
                    self.NeighConsensus.state_dict()[name].copy_(
                        checkpoint['state_dict']['NeighConsensus.' + name])

            print('Done!')

        self.FeatureExtraction.eval()

        if self.half_precision:
            for p in self.NeighConsensus.parameters():
                p.data = p.data.half()

            for l in self.NeighConsensus.conv:
                if isinstance(l, Conv4d):
                    l.use_half = True

    # used only for forward pass at eval and for training with strong supervision
    def forward(self, tnf_batch):
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])

        feature_A2 = featureL2Norm(feature_A[1])     # 2nd level in fpn feature. one 16th of original image size
        feature_B2 = featureL2Norm(feature_B[1])
        feature_A0 = featureL2Norm(feature_A[0])     # 0th level in fpn feature. one fourth of original image size
        feature_B0 = featureL2Norm(feature_B[0])

        # use 2nd level feature to form 4d corr
        if self.half_precision:
            feature_A2 = feature_A2.half()
            feature_B2 = feature_B2.half()


        # feature correlation
        corr4d = self.FeatureCorrelation(feature_A2, feature_B2)


        # run match processing model
        corr4d = MutualMatching(corr4d)

        corr4d = self.NeighConsensus(corr4d)

        corr4d = MutualMatching(corr4d)

        return corr4d, feature_A0, feature_B0


