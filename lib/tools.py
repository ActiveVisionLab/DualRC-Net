import torch.nn.functional as F
import os, time, sys, math
from lib.model_v2 import ImMatchNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import lib.interpolator as interpolator

def calc_gt_indices(batch_keys_gt ,batch_assignments_gt):
    """
        calc_gt_indices() calculate the ground truth indices and number of valid key points.
        Arguments:
            batch_keys_gt: [tensor B x N x 3] the last column stores the indicator of the key points
                            if it is larger than 0 it is valid, otherwise invalid
            batch_assignments_gt [tensor B x N x N], the ground truth assignment matrix
        Returns:
            indices_gt: [tensor B x N ]: from source to target, each element stores the index of target matches
            key_num_gt: [tensor B]: number of valid key points per input of the batch
    """
    _, indices_gt = torch.max(batch_assignments_gt, 2) # get ground truth matches from source to target
    indices_gt += 1 # remember that indices start counting from 1 for 0 is used to store empty key points 
    mask_gt = (batch_keys_gt[:,:,2]>0).long() # get the valid key point masks
    indices_gt = indices_gt * mask_gt
    key_num_gt = mask_gt.sum(dim=1).float()
    return indices_gt, key_num_gt


class ExtractFeatureMap:
    def __init__(self, im_fe_ratio, use_cuda = True, device=None):
        # im_fe_ratio, the ratio between the input image resolution to feature map resolution
        self.im_fe_ratio = im_fe_ratio
        # interp [object of Interpolator]: to interpolate the correlation maps
        self.use_cuda = use_cuda
        self.interp = interpolator.Interpolator(im_fe_ratio, device)
        self.device = device
    def normalise_per_row(self, keycorr):
        """
        normalise_per_row() normalise the 3rd dimension by calculating its sum and divide the vector
                in last dimension by the sum
        Arguments
            keycorr: B x N x HW
        Returns
            keycorr: B x N x HW
        """
        eps = 1e-15
        sum_per_row = keycorr.sum(dim=2, keepdim=True) + eps
        sum_per_row = sum_per_row.expand_as(keycorr)  # B x N x L
        keycorr = keycorr / sum_per_row

        return keycorr

    def __call__(self, corr, query_keypoints, source_to_target=True):
        """
        extract_featuremap() extract the interpolated feature map for each query key points in query_keypoints
        Arguements
            corr [tensor float] B x 1 x H1 x W1 x H2 x W2: the 4d correlation map
            query_keypoints [tensor float] B x N x 2: the tensor stores the sparse query key points
            source_to_targe [boolean]: if true, query from source to target, otherwise, from target to source
        Return:
            keycorr [tensor float]: B x N x H2 x W2 (when source_to_targe = True) the correlation map for each source
                                    key ponit
        """
        B, C, H1, W1, H2, W2 = corr.shape
        _, N, _ = query_keypoints.shape
        if source_to_target:
            corr = corr.view(B, H1, W1, H2 * W2)
            corr = corr.permute(0, 3, 1, 2)
            keycorr = self.interp(corr, query_keypoints, H2, W2)  # keycorr B x H2*W2 x N, key is source
            keycorr = keycorr.permute(0, 2, 1)  # B x N x H2*W2
            keycorr = keycorr.view(B, N, H2, W2)  # B x N x H2 x W2
            H, W = H2, W2
        else:
            corr = corr.view(B, H1 * W1, H2, W2)
            keycorr = self.interp(corr, query_keypoints, H1, W1)  # keycorr B x H1*W1 x N query_keypoints is target point
            keycorr = keycorr.permute(0, 2, 1)  # B x N x H1*W1
            keycorr = keycorr.view(B, N, H1, W1)  # B x N x H1 x W1
            H, W = H1, W1

        keycorr = keycorr.view(B, N, -1).contiguous()
        # try softmax
        # keycorr = torch.softmax(keycorr, dim = 2)
        keycorr = self.normalise_per_row(keycorr)
        return keycorr.view(B, N, H, W)

    def selected_corr_to_matches(self, keycorr):
        """
            selected_corr_to_matches() find the best matches for each row in keycorr
            keycorr [tensor float]: B x N x H x W (when source_to_targe = True) the correlation map for each source
                                    key ponit, note H x W must be aligned with the original image size
            Returns
                xyA [tensor float]: B x N x 2, the key points from source to target
                score[tensor float]: B x N, the score of the key points
        """

        B, N, H, W = keycorr.shape
        start = time.time()
        # XA, YA = np.meshgrid(range(W), range(H))  # pixel coordinate
        # XA, YA = torch.FloatTensor(XA).view(-1).cuda(self.device), torch.FloatTensor(YA).view(-1).cuda(self.device)

        keycorr = keycorr.view(B, N, -1)

        score, indices = torch.max(keycorr, dim=2)
        yA = torch.floor(torch.div(indices, W).float())
        xA = (indices.float() - yA*W)
        xA = xA.view(B, N, 1)
        yA = yA.view(B, N, 1)

        # xA = XA[indices.view(-1)].view(B, N, 1)
        # yA = YA[indices.view(-1)].view(B, N, 1)
        xyA = torch.cat((xA, yA), 2)
        xyA *= self.im_fe_ratio
        end = time.time()
        return xyA, score, end-start

    def regularise_corr(self, corr,source_to_target = True):
        B, _, H1, W1, H2, W2 = corr.shape

        if source_to_target:
            corr = corr.view(B, H1, W1, H2*W2)
            ncorr =torch.nn.functional.softmax(corr,dim=3)
        else:
            corr = corr.view(B, H1*W1, H2,W2).permute(0,2,3,1).contiguous() 
            ncorr =torch.nn.functional.softmax(corr,dim=3)

        dx = ncorr[:,:,:-1,:] - ncorr[:,:,1:,:]    
        dy = ncorr[:,:-1,:,:] - ncorr[:,1:,:,:]    
        loss = dx[:, :-1, :] * dx[:, :-1, :] + dy[:, :, :-1] * dy[:, :, :-1]
        loss = loss.sum(dim=3)
        loss = torch.sqrt(loss)   
        return loss.sum()/B


class ImgMatcher:
    '''
    This is the class that integrate the model and all postprocessing part together
    '''
    def __init__(self, checkpoint, use_cuda, half_precision, im_fe_ratio = 16, postprocess_device = None):

        checkpoint_ = torch.load(checkpoint)
        multi_gpu = checkpoint_['args'].multi_gpu

        self.model = ImMatchNet(use_cuda=use_cuda, multi_gpu=multi_gpu, half_precision=half_precision, checkpoint=checkpoint)
        self.model.eval()
        self.feature_extractor = ExtractFeatureMap(im_fe_ratio=self.im_fe_ratio, device=postprocess_device)
        self.postprocess_device = postprocess_device
        self.im_fe_ratio = im_fe_ratio

    def __call__(self, batch, num_pts=2000, iter_step=1000, central_align = True, *args, **kwargs):
        '''
        Input:
            batch:
                [dict] {'source_image': [tensor float] B x C x H1 x W1, 'target_image': [tensor float] B x C x H2 x W2}
            num_pts:
                [int] number of matches generated.
            iter_step:
                [int] step when calculate the matches. Leave it fixed.
            central_align:
                [bool] whether to centrally align the feature map and image.
                using coarse-resolution 4D corr to select preliminary region.
        Output
            matches:
                [tensor float] B x N x 2 matches in image scale
            scores:
                [tensor float] B x N
            output:
                [tuple tensor] export corr4d, featureA_0, featureB_0
        '''
        batch['source_image'] = batch['source_image'].cuda()
        batch['target_image'] = batch['target_image'].cuda()

        output = self.model(batch)
        corr4d, feature_A0, feature_B0 = output
        # move output to the gpu of feature_extractor if two things are not on the same device
        if torch.cuda.current_device() != self.feature_extractor.device:
            start = time.time()
            corr4d = corr4d.cuda(self.feature_extractor.device)
            feature_A0 = feature_A0.cuda(self.feature_extractor.device)
            feature_B0 = feature_B0.cuda(self.feature_extractor.device)
            end = time.time()
            print('Time for transferring tensor to a different GPU %f second'%(end-start))
        output = (corr4d, feature_A0, feature_B0)
        self.high_low_ratio = int(np.round(feature_A0.shape[2] / corr4d.shape[2]))

        # use corr4d as a pre-processing filter
        query_src = self.generate_query_from_corr4d(corr4d)
        query_src = query_src.cuda(self.feature_extractor.device)

        matches_src_to_ref, scores_src_to_ref = self.find_matches(output, query_src,
                                                                  source_to_target=True,
                                                                  iter_step=iter_step)  # B x H*W x 2 in image coordinate
        matches_src_to_ref_to_src, scores_src_to_ref_to_src = self.find_matches(output, matches_src_to_ref,
                                                                                source_to_target=False,
                                                                                iter_step=iter_step)  # B x H*W x 2 in image coordinate

        # pick out the mutual neighbour
        src_to_ref = torch.cat((query_src, matches_src_to_ref), dim=2)
        ref_to_src = torch.cat((matches_src_to_ref_to_src, matches_src_to_ref), dim=2)
        src_to_ref = src_to_ref / (self.im_fe_ratio / self.high_low_ratio)  # convert into feature coordinate
        ref_to_src = ref_to_src / (self.im_fe_ratio / self.high_low_ratio)

        result = src_to_ref - ref_to_src
        result = torch.norm(result, dim=2)
        matches = src_to_ref[result == 0]
        scores_src_to_ref = scores_src_to_ref[result == 0]
        scores_src_to_ref_to_src = scores_src_to_ref_to_src[result == 0]
        scores = scores_src_to_ref_to_src + scores_src_to_ref

        # only select the first 'num_pts
        scores, indice = torch.sort(scores, descending=True)
        matches = matches[indice]
        matches = matches[:num_pts, :]
        scores = scores[:num_pts]
        print('Number of matches output', scores.shape[0])

        if central_align:
            matches = matches + 0.5

        matches = matches * (self.im_fe_ratio / self.high_low_ratio)  # convert to image coordinate

        return matches, scores, output


    def make_grid(self, B, H, W):
        x = torch.linspace(0, W-1, steps=W)
        y = torch.linspace(0, H-1, steps=H)

        x, y = torch.meshgrid(x, y)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        xy = torch.cat((x, y), dim=1)
        xy = xy.expand((B, -1, -1))
        return xy

    def find_matches(self, output, query, source_to_target, iter_step):
        corr4d, feature_A0, feature_B0 = output
        ratio = self.im_fe_ratio / self.high_low_ratio

        N = query.shape[1]
        step = iter_step
        match_list = []
        score_list = []
        query_list = [query[:, i: i + step, :] for i in range(0, N - step, step)]
        if N % step == 0:
            query_list.append(query[:, N - step: N, :])
        else:
            query_list.append(query[:, N - (N % step): N, :])

        for idx in range(len(query_list)):
            query_ = query_list[idx]
            num = query_.shape[1]

            tgtCorr_ = self.feature_extractor(corr4d.float(), query_.float(), source_to_target=source_to_target)

            if source_to_target:
                tgtCorr_lv0 = sparse_feature_correlation(query_, ratio, feature_A0, feature_B0)
            else:
                tgtCorr_lv0 = sparse_feature_correlation(query_, ratio, feature_B0, feature_A0)

            tgtCorr_ = mask_over_corr(tgtCorr_, tgtCorr_lv0)

            matches_, score_, time_ = self.feature_extractor.selected_corr_to_matches(tgtCorr_)

            matches_ = matches_ / self.high_low_ratio

            match_list.append(matches_)
            score_list.append(score_)

        matches = torch.cat(match_list, dim=1)
        scores = torch.cat(score_list, dim=1)
        return matches, scores

    def generate_query_from_corr4d(self, corr4d, thres=0.5):
        '''
        Use corr4d as a pre-processing filter to generate query which is to be fed into high resolution feature map
        '''
        B, C, h1, w1, h2, w2 = corr4d.shape
        num = h1*w1
        num = int(np.floor(num*thres))
        corr4d_max, _ = torch.max(corr4d.view(B, C, h1, w1, -1), dim=4)
        # print(corr4d_max)
        _, indice = torch.sort(corr4d_max.view(B, C, -1), dim=2, descending=True)
        indice = indice[:, :, :num]
        y = torch.floor(torch.div(indice, w1).float())
        x = (indice.float() - y*w1)
        y = y * self.high_low_ratio
        x = x * self.high_low_ratio
        x_, y_ = [], []
        for i in range(self.high_low_ratio):
            for j in range(self.high_low_ratio):
                x_.append(x+j)
                y_.append(y+i)
        x = torch.cat(x_, dim=2).unsqueeze(3)
        y = torch.cat(y_, dim=2).unsqueeze(3)
        xy = torch.cat((x, y), dim=3).squeeze(0)

        return xy * (self.im_fe_ratio / self.high_low_ratio)


def sparse_feature_correlation(pts_A, im_fe_ratio, feature_A, feature_B, B_candidate=None):
    '''
    Compute feature correlation between some keypoints (pts_A) in feature_A w.r.t entire feature map feature_B
    pts_A [tensor float]: B x N x 2 the keypoint location in image
    feature_A [tensor float]: B x C x H1 x W1 the feature map of image A
    feature_B [tensor float]: B x C x H2 x W2 the feature map of image B
    '''
    B, C, H1, W1 = feature_A.shape
    _, _, H2, W2 = feature_B.shape

    # extract feature out of feature_A
    pts_A = pts_A / im_fe_ratio
    pts_A = torch.round(pts_A).long()

    pts_A[:, :, 0][pts_A[:, :, 0] > W1-1] = W1 - 1   # prevent some indice exceeding the range
    pts_A[:, :, 1][pts_A[:, :, 1] > H1-1] = H1 - 1

    if B_candidate is not None:
        B_candidate[:, :, :, 0][B_candidate[:, :, :, 0] > W2 - 1] = W2 - 1
        B_candidate[:, :, :, 1][B_candidate[:, :, :, 1] > H2 - 1] = H2 - 1
        B_candidate = B_candidate[:, :, :, 1] * W2 + B_candidate[:, :, :, 0]

    indice_A = pts_A[:, :, 1] * W1 + pts_A[:, :, 0]     # B x N
    indice_A = indice_A.unsqueeze(1).expand(B, C, -1)    # B x C x N
    feature_pts = torch.gather(feature_A.reshape(B, C, -1), 2, indice_A)     # B x C x N

    # compute the feature correlation (dot product)
    if B_candidate is None:
        feature_corr = torch.bmm(feature_pts.transpose(1, 2), feature_B.view(B, C, -1))     # B x N x (H2.W2)
        return feature_corr.view(B, -1, H2, W2)
    else:
        B, N, Nc = B_candidate.shape
        feature_pts = feature_pts.permute(2, 0, 1)  # N x 1 x C
        B_candidate = B_candidate.expand(C, -1, -1).permute(1, 0, 2)    # N x C x Nc
        feature_candidate = torch.gather(feature_B.view(B, C, -1).expand(N, -1, -1), 2, B_candidate)
        feature_corr = torch.bmm(feature_pts, feature_candidate)
        return feature_corr.squeeze(1)      # N x Nc


def mask_over_corr(fe_corr_A, fe_corr_B):
    '''
    Mask fe_corr_A over fe_corr_B, where fe_corr_A is the correlation after going through Neighbour Consensus Network
    and it should be smaller than fe_corr_B.
    fe_corr_A [tensor float] B x N x HA x WA feature correlation score of keypoint obtained from NCN
    fe_corr_B [tensor float] B x N x HB x WB feature correlation score of keypoint obtained directly from larger feature map.
    '''
    B, N, HA, WA = fe_corr_A.shape
    _, _, HB, WB = fe_corr_B.shape
    ratio = np.round(HB / HA)
    ratio = int(ratio)
    # use softmax to turn the mask into a probability matrix
    # fe_corr_A = F.softmax(fe_corr_A.view(B, N, -1), dim=2).view(B, N, HA, WA)
    # expand the fe_corr_A
    # fe_corr_A_ = torch.repeat_interleave(fe_corr_A, ratio, dim=2)
    # fe_corr_A_ = torch.repeat_interleave(fe_corr_A_, ratio, dim=3)

    fe_corr_A_ = F.interpolate(fe_corr_A, mode='nearest', size=(HB, WB))
    
    # make sure that fe_corr_A_ has the same size as fe_corr_B
    fe_corr_A_ = fe_corr_A_[:, :, :HB, :WB]
    fe_corr_B_ = fe_corr_A_ * fe_corr_B
    return fe_corr_B_


def UnNormalize(img, mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    img = img*std + mean
    return img


def pre_draw(img):
    img = img.squeeze(0).cpu()
    img = UnNormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.permute([1, 2, 0])
    return img


def draw_lines_for_matches(src, tgt, pts1, pts2):
    gap = 20
    fig = plt.figure(figsize=(12, 21))
    h1, w1, _ = src.shape
    h2, w2, _ = tgt.shape
    h = h1 + h2 + gap
    w = np.max((w1, w2))
    img = np.ones((h, w, 3))
    img[:h1, :w1, :] = src
    img[h1 + gap:h1 + gap +h2, :w2, :] = tgt

    pts2[:, 1] = pts2[:, 1] + gap + h1
    plt.imshow(img)
    for pts in zip(pts1, pts2):
        plt.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], 'o--g', linewidth=1, markersize=4)
    plt.axis('off')
