from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Interpolator(nn.Module):
    def __init__(self, im_fe_ratio, device = None):
        super(Interpolator, self).__init__()
        self.im_fe_ratio = im_fe_ratio
        self.device = device
        self.maxXY = torch.ones(1, 1, 2)
        self.minXY = torch.zeros(1, 1, 2)

    def getMaxMinXY(self, B, N, H, W):
        # B1, N1, _ = self.maxXY.shape
        # B2, N2, _ = self.minXY.shape
        # if B1 == B and N1 == N and B2 == B and N2 == N:
        #     return self.maxXY, self.minXY
        # else:
        self.maxXY = torch.ones(1, 1, 2)
        self.minXY = torch.zeros(1, 1, 2)
        self.maxXY[0, 0, 0] = (W - 1)
        self.maxXY[0, 0, 1] = (H - 1)
        self.maxXY = self.maxXY.cuda(self.device)
        self.minXY = self.minXY.cuda(self.device)
        self.maxXY = self.maxXY.expand(B, N, -1)  # B x N x 2
        self.minXY = self.minXY.expand(B, N, -1)  # B x N x 2
        return self.maxXY, self.minXY

    def maskoff(self, feature_per_kp, keypoints):
        """
            maskoff() set the features to be zeros if the keypoints are 0, 0
        Arguments:
            feature_per_kp [float tensor] B x C X N : standard feature tensor by number of key points
            keypoints [float tensor] B x N x 2: key points
        Returns:
            feature_per_kp [float tensor] B x C x N : standard feature tensor with invalid keypoints
        """
        mask = (keypoints[:, :, :2] > 1e-10).float().mean(dim=2)  # B x N
        mask = mask.unsqueeze(1)  # B x 1 x N
        mask = mask.expand_as(feature_per_kp)  # B x C x N
        feature_per_kp = feature_per_kp * mask
        return feature_per_kp

    def forward(self, feature, keypoints, Hf=None, Wf=None):
        """
        Interpolator(): collects a set of sparse key points by interpolating from
                       the feature map.
        Arguments
            feature [float tensor] B x C x H x W: standard feature map
            keypoints [float tensor] B x N x 2: key points
        Return
            UF [float tensor] B x C x N: the sparse interpolated features collected
                                         at the input sparse key point locations.
                                         note that the rows corresponding to invalid key points
                                         are masked off as zeros.
        """
        B, C, H, W = feature.shape
        feature = feature.view(B, C, -1)  # B x C x HW, B x 1024 x 1024
        # print('keypoints', keypoints[0,1])
        # convert the key points from the image coordinate to feature map coordinate
        keypoints = keypoints / self.im_fe_ratio
        _, N, _ = keypoints.shape

        # print(keypoints[0])
        maxXY, minXY = self.getMaxMinXY(B, N, H, W)

        # nearest neighbours
        iLower = torch.max(torch.floor(keypoints), minXY)  # B x N x 2, index of X, Y
        # iLower = torch.floor(keypoints)   # B x N x 2, index of X, Y
        iUpper = torch.min(torch.ceil(keypoints), maxXY)
        # iUpper = torch.ceil(keypoints)
        upper = keypoints - iLower  # note that weight is the 1 - distance
        lower = 1 - upper  # B x N x 2

        iX = torch.cat((iLower[:, :, 0].unsqueeze(2), iUpper[:, :, 0].unsqueeze(2)), 2)
        iY = torch.cat((iLower[:, :, 1].unsqueeze(2), iUpper[:, :, 1].unsqueeze(2)), 2)
        xX = torch.cat((lower[:, :, 0].unsqueeze(2), upper[:, :, 0].unsqueeze(2)), 2)
        yY = torch.cat((lower[:, :, 1].unsqueeze(2), upper[:, :, 1].unsqueeze(2)), 2)

        iX = iX.unsqueeze(2).expand(-1, -1, 2, -1).long()  # B x 32 x 2 x 2 ( x0 x1; x0 x1 )
        iY = iY.unsqueeze(2).expand(-1, -1, 2, -1).transpose(2, 3).long()
        xX = xX.unsqueeze(2).expand(-1, -1, 2, -1)
        yY = yY.unsqueeze(2).expand(-1, -1, 2, -1).transpose(2, 3)

        iX = iX.view(B, N, -1)  # B x N x 4
        iY = iY.view(B, N, -1)
        xX = xX.contiguous().view(B, N, -1)
        yY = yY.contiguous().view(B, N, -1)
        # print('iY', iY[0,1])
        # print('iX', iX[0,1])
        # print('xY', yY[0,1])
        # print('xX', xX[0,1])
        # print('xY*xY', (xX*yY)[0,1])
        coeff = (xX * yY).contiguous().view(B, -1)  # B x N*4
        # print('coeff', coeff[0,:8])
        coeff = coeff.unsqueeze(dim=1).expand(-1, C, -1)  # B x C x N*4

        # print('H', H, 'W', W)
        indices = (iY * W + iX).view(B, N * 4)  # B x N*4
        # print('indices', indices[0,0:8])
        indices = indices.unsqueeze(dim=1).expand(-1, C, -1)  # B x C x N*4
        # print('2.indices', indices[0, 0, :8], indices[0, 1, :8])
        UF = torch.gather(feature, 2, indices)  # B x C x N*4       
        # -> B x C x B x 4
        # interpolation here -> 


        # UF *= self.im_fe_ratio
        UF = (UF * coeff)  # B x
        # np.savetxt('UF', UF[0,:,:8].detach().cpu().numpy() )
        UF = UF.reshape(B, C, N, -1)
        # np.savetxt('UF2', UF[0,:,1,:].detach().cpu().numpy() )
        UF = UF.sum(dim=3)  # B x C x N
        # print('UF',UF.shape)
        UF = self.maskoff(UF, keypoints)
        # print('UF', UF.shape)
        # print('UF', UF.shape)
        return UF


class LocationInterpolator(nn.Module):
    def __init__(self, im_fe_ratio, device=None):
        super(LocationInterpolator, self).__init__()
        self.interpolator = Interpolator(im_fe_ratio)
        self.device = device
    def forward(self, ijB_A, keypoints):
        """
        LocationInterpolator() is to collect a set of interpolated correspondence pixel
                               locations
        Arguments:
            ijB_A [long tensor]: B x 2 x H x W : is the tensor storing the 2D pixel
                                    locations from source image A to targe image B
            keypoints [float tensor] B x N x 2: key points
        Return:
            xyB_A [float tensor]: B x N x 2 the interpolated correspondnce map for the set of sparse
                                 key points.
                                 note that the rows corresponding to invalid key points
                                         are masked off as zeros.
        """
        xyB_A = self.interpolator(ijB_A.float(), keypoints) * self.interpolator.im_fe_ratio
        return xyB_A.transpose(2, 1)


class InverInterpolator(Interpolator):
    def __init__(self, im_fe_ratio, kernel_size=5, N=32, mode=1, device=None):
        super(InverInterpolator, self).__init__(im_fe_ratio)
        self.device = device
        self.kernel_size = kernel_size
        self.mode = mode
        if kernel_size > 0:
            # add gaussian
            gaussian_filter = nn.Conv2d(in_channels=N, out_channels=N, padding_mode='zeros',
                                        padding=(int(kernel_size / 2), int(kernel_size / 2)),
                                        kernel_size=kernel_size, groups=N, bias=False)
            if kernel_size == 3:
                gk = torch.FloatTensor(
                    np.array([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]]))
            elif kernel_size == 5:
                gk = torch.FloatTensor(np.array(
                    [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                     [1, 4, 7, 4, 1]])) / 273.
            elif kernel_size == 7:
                gk = torch.FloatTensor(np.array(
                    [[0, 0, 0, 5, 0, 0, 0], [0, 5, 18, 32, 18, 5, 0], [0, 18, 64, 100, 64, 18, 0],
                     [5, 32, 100, 100, 100, 32, 5], [0, 18, 64, 100, 64, 18, 0], [0, 5, 18, 32, 18, 5, 0],
                     [0, 0, 0, 5, 0, 0, 0]]))
                gk /= gk.sum()

            gk = gk.unsqueeze(0).unsqueeze(1)
            gk = gk.expand(N, -1, -1, -1)
            gaussian_filter.weight.data = gk
            gaussian_filter.weight.requires_grad = False
            self.gaussian_filter = gaussian_filter.cuda(self.device)

    def get_1nn(self, Xg, keypoint_g, H, W):
        """
        Arguments:
            Xg [tensor] B x N x N
            keypoint_g [tensor] B x N x 2
            H height, resolution of H and W
            W width
        Return
            onehot [tensor] B x N x HW
        """
        B, N, _ = keypoint_g.shape
        xyGt = torch.bmm(Xg, keypoint_g) / self.im_fe_ratio - 0.5  # B x N x 2 float gt coordinate in feature map
        maxXY, minXY = self.getMaxMinXY(B, N, H, W)

        boundedXY = torch.max(xyGt, minXY)  # B x N x 2, index of X, Y in feature map
        boundedXY = torch.min(boundedXY, maxXY)  # B x N x 2, index of X, Y in feature map

        boundedXY = boundedXY.long()
        indices = boundedXY[:, :, 1] * W + boundedXY[:, :, 0]  # B x N x 1
        indices = indices.unsqueeze(2)
        coeff = torch.ones(B, N, 1).cuda(self.device)
        onehot = torch.zeros(B, N, H * W).cuda(self.device)
        onehot.scatter_(dim=2, index=indices, src=coeff)
        if self.kernel_size > 0:
            onehot = self.gaussian_filter(onehot.view(B, N, H, W)).view(B, N, H * W)  # add gaussian blur
        mask = Xg.sum(dim=2, keepdim=True).expand_as(onehot)  # B x N x HW
        onehot *= mask

        # for n in range(N):
        #     print(xyGt[0][n])
        #     plt.imshow(onehot[0][n].cpu().view(H,W))
        #     plt.show()
        return onehot

    def get_4nn(self, Xg, keypoint_g, H, W):
        """
        Arguments:
            Xg [tensor] B x N x N
            keypoint_g [tensor] B x N x 2
            H height, resolution of H and W
            W width
        Return
            onehot [tensor] B x N x HW
        """
        # convert into feature map coordinate
        B, N, _ = keypoint_g.shape
        xyGt = torch.bmm(Xg, keypoint_g) / self.im_fe_ratio - 0.5  # B x N x 2 float gt coordinate in feature map
        maxXY, minXY = self.getMaxMinXY(B, N, H, W)

        # nearest neighbours
        iLower = torch.max(torch.floor(xyGt), minXY)  # B x N x 2, index of X, Y in feature map
        iUpper = torch.min(torch.ceil(xyGt), maxXY)  # B x N x 2,
        upper = xyGt - iLower  # note that weight is the 1 - distance
        lower = 1 - upper  # B x N x 2

        iX = torch.cat((iLower[:, :, 0].unsqueeze(2), iUpper[:, :, 0].unsqueeze(2)), 2)
        iY = torch.cat((iLower[:, :, 1].unsqueeze(2), iUpper[:, :, 1].unsqueeze(2)), 2)
        xX = torch.cat((lower[:, :, 0].unsqueeze(2), upper[:, :, 0].unsqueeze(2)), 2)
        yY = torch.cat((lower[:, :, 1].unsqueeze(2), upper[:, :, 1].unsqueeze(2)), 2)

        iX = iX.unsqueeze(2).expand(-1, -1, 2, -1).long()  # B x 32 x 2 x 2 ( x0 x1; x0 x1 )
        iY = iY.unsqueeze(2).expand(-1, -1, 2, -1).transpose(2, 3).long()
        xX = xX.unsqueeze(2).expand(-1, -1, 2, -1)
        yY = yY.unsqueeze(2).expand(-1, -1, 2, -1).transpose(2, 3)

        iX = iX.view(B, N, -1)  # B x N x 4
        iY = iY.view(B, N, -1)
        xX = xX.contiguous().view(B, N, -1)
        yY = yY.contiguous().view(B, N, -1)

        coeff = (xX * yY).contiguous()  # B x N x 4
        indices = iY * W + iX  # B x N x4

        onehot0 = torch.zeros(B, N, H * W).cuda(self.device)
        onehot0.scatter_(dim=2, index=indices, src=coeff)


        mask = Xg.sum(dim=2, keepdim=True).expand_as(onehot0)  # B x N x HW make none key points 0
        onehot0 *= mask
        if self.kernel_size > 0:
            onehot1 = self.gaussian_filter(onehot0.view(B, N, H, W)).view(B, N, H * W)  # add gaussian blur
            onehot2 = self.gaussian_filter(onehot1.view(B, N, H, W)).view(B, N, H * W)  # add gaussian blur
            onehot1 *= mask
            onehot2 *= mask

            # for n in range(N):
        #     print(xyGt[0][n])
        #     plt.imshow(onehot[0][n].cpu().view(H,W))
        #     plt.show()
        return onehot2, onehot1, onehot0

    def forward(self, Xg, keypoint_g, H, W):
        """
        Arguments:
            Xg [tensor] B x N x N
            keypoint_g [tensor] B x N x 2
            H height, resolution of H and W  
            W width 
        Return
            onehot [tensor] B x N x HW    
        """
        if self.mode == 0:
            return self.get_1nn(Xg, keypoint_g, H, W)
        elif self.mode == 1:
            return self.get_4nn(Xg, keypoint_g, H, W)[1]
        elif self.mode == 2:
            return self.get_4nn(Xg, keypoint_g, H, W)[0]
        elif self.mode == 3:
            return self.get_4nn(Xg, keypoint_g, H, W)

        return self.get_4nn(Xg, keypoint_g, H, W)[1]

