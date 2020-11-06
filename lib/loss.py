'''
This script tends to enforce strong loss of training
'''
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from lib import constant
from lib import tools
from lib import interpolator

class SparseLoss(nn.Module):
    def __init__(self, image_size, model, loss_name='mse',
                 weight_orthogonal=0.1):
        super(SparseLoss, self).__init__()
        self.image_size = image_size
        self.model = model
        self.criterions = {'weighted_mse': self.weighted_mse_loss,
                           'fnorm': self.fnorm_loss,
                           'meanfnorm': self.mean_fnorm_loss,
                           'orthogonal_meanfnorm': self.orthogonal_meanfnorm,
                           'displacement': self.displacement_loss,
                           'balanced': self.balanced_loss,
                           'combined': self.combined_loss,
                           }
        self.loss_fn = self.criterions[loss_name]
        self.weight_orthogonal = weight_orthogonal

    def calc_weight(self, target):
        batch_size = target.shape[0]
        Weight = torch.ones(target.shape)
        for b in range(batch_size):
            XGT = target[b]
            K = float(XGT.sum())  # K is the number of positive
            Wp = 0.5 / K  # positive weights
            Wn = 0.5 / (225 - K)  # negative weights
            Weight[b] *= Wn
            Weight[b, XGT > 0.5] = Wp
        return Weight.cuda()

    def weighted_mse_loss(self, prediction, data_batch):
        target = data_batch['assignment']
        weight = self.calc_weight(target)
        return torch.mean(weight * (prediction - target) ** 2)

    def fnorm_loss(self, prediction, data_batch):
        target = data_batch['assignment']
        loss = prediction - target
        return torch.norm(loss)

    def mean_fnorm_loss(self, prediction, data_batch):
        target = data_batch['assignment']
        B = target.shape[0]
        loss = (prediction - target).view(B, -1).norm(dim=1).mean()
        return loss

    def binormalise(self, X):
        for i in range(1):
            X = X * X
            X_sum_2 = torch.sum(X, dim=2, keepdim=True)  # normalisation
            X = X / (X_sum_2 + constant._eps)
            X_sum_1 = torch.sum(X, dim=1, keepdim=True)  # normalisation
            X = X / (X_sum_1 + constant._eps)
        return X

    def orthogonal(self, X, data_batch):
        X = self.binormalise(X)
        Xg = data_batch['assignment']
        Xgt = Xg.permute(0, 2, 1)
        Xt = X.permute(0, 2, 1)

        # source to target
        Xo = torch.bmm(X, Xt)
        Xog = torch.bmm(Xg, Xgt)
        Dif = (Xo - Xog)  # .diagonal(dim1=1, dim2=2) # Trace ( Xo - Xg ) #????

        # target to source
        Xo2 = torch.bmm(Xt, X)
        Xog2 = torch.bmm(Xgt, Xg)
        Dif2 = (Xo2 - Xog2)  # .diagonal(dim1=1, dim2=2) # Trace ( Xo - Xg )
        loss = (torch.norm(Dif, dim=1) + torch.norm(Dif2, dim=1))
        return loss.mean() / 64

    def orthogonal_meanfnorm(self, prediction, data_batch):
        mfn_loss = self.mean_fnorm_loss(prediction, data_batch)
        orth_loss = self.orthogonal(prediction, data_batch)
        loss = (1 - self.weight_orthogonal) * mfn_loss + self.weight_orthogonal * orth_loss
        return loss

    def displacement_loss(self, prediction, data_batch):
        # source to target
        key1 = data_batch['target_points']
        Xg = data_batch['assignment']
        B = Xg.shape[0]
        dif = prediction - Xg
        key_dif = torch.bmm(dif, key1[:, :, :2])
        loss1 = key_dif.view(B, -1).norm(dim=1).mean()
        # target to source
        key2 = data_batch['source_points']
        dif2 = dif.permute(0, 2, 1)
        key_dif2 = torch.bmm(dif2, key2[:, :, :2])
        loss2 = key_dif2.view(B, -1).norm(dim=1).mean()

        return (loss1 + loss2) / self.image_size

    def balanced_loss(self, prediction, data_batch):
        # source to target
        Xg = data_batch['assignment']
        B = Xg.shape[0]
        mask = torch.gt(Xg, 0)
        # positive
        dif = prediction - Xg  # prediction is after relu, and Xg is binary
        positive_loss = 0
        for b in range(B):
            positive = torch.masked_select(dif[b], mask[b])  # B * source_key_num_gt
            positive_loss += positive.norm()
        positive_loss /= B
        # negative
        dif2 = dif.clone()
        dif2[mask] = 0.
        negative, _ = torch.max(dif2, dim=2)  # B x source_key_num_gt
        negative_loss = negative.norm(dim=1).mean()

        # to implement the targe to source loss
        negative2, _ = torch.max(dif2, dim=1)  # B x source_key_num_gt
        negative_loss2 = negative2.norm(dim=1).mean()

        return positive_loss + (negative_loss + negative_loss2) * 0.5

    def combined_loss(self, prediction, data_batch):
        o_loss = self.orthogonal(prediction, data_batch)
        d_loss = self.displacement_loss(prediction, data_batch)
        b_loss = self.balanced_loss(prediction, data_batch)
        return (1 - self.weight_orthogonal) * d_loss + self.weight_orthogonal * o_loss + b_loss, \
               d_loss, o_loss, b_loss

    def forward(self, data_batch):
        X, _, _ = self.model(data_batch)
        return self.loss_fn(X, data_batch), X


class WeakLoss(SparseLoss):
    def __init__(self, image_size, model, loss_name='meanfnorm',
                 weight_orthogonal=0.1, with_sparseloss=False, use_cuda=True):
        super(WeakLoss, self).__init__(image_size, model, loss_name=loss_name, weight_orthogonal=weight_orthogonal)
        self.with_sparseloss = with_sparseloss
        self.I = torch.eye(4).cuda()
        self.with_softmax = True

    def normalised_max_correlation(self, corr4d):
        """
        normalised_max_correlation() takes in the 4D correlation map and returns a score to measure the pair of the image of similarity
        Argument:
            corr4d [tensor] B x 1 x Ha x Wa x Hb x Wb float.
        Return:
            score [tensor] float 1x1 , the batch of correlation scores.
            iB [tensor] B, the index of max correlation
            iA [tensor] B, the same as iB but with inverse direction
        """
        B, C, Ha, Wa, Hb, Wb = corr4d.shape

        nc_B_Avec = corr4d.view(B, Ha * Wa, Hb, Wb)  # B x 256 x 16 x 16
        nc_A_Bvec = corr4d.view(B, Ha, Wa, Hb * Wb).permute(0, 3, 1, 2)  # same as above

        if self.with_softmax:
            nc_B_Avec = nn.functional.softmax(nc_B_Avec, 1)  # normalised per 256
            nc_A_Bvec = nn.functional.softmax(nc_A_Bvec, 1)
        else:
            nc_B_Avec = tools.NormalisationPerRow(nc_B_Avec)
            nc_A_Bvec = tools.NormalisationPerRow(nc_A_Bvec)

        # compute matching scores
        scores_B, _ = torch.max(nc_B_Avec, dim=1)  # B x 16 x 16
        scores_A, _ = torch.max(nc_A_Bvec, dim=1)

        # scoresB_rank = scores_B.view(B, -1 ) # B x 256
        # scoresA_rank = scores_A.view(B, -1 ) # B x 256
        # _, indicesB = torch.sort(scoresB_rank) # B x 256
        # _, indicesA = torch.sort(scoresA_rank) # B x 256

        score = torch.mean(scores_A + scores_B) / 2

        return score  # , indicesB[:,:4] , indicesA[:,:4]

    def weak_supervision(self, corr, tnf_batch):
        # positive
        score_pos = self.normalised_max_correlation(corr)

        # negative
        b = tnf_batch['source_image'].size(0)
        tnf_batch['source_image'] = tnf_batch['source_image'][np.roll(np.arange(b), -1), :]  # roll
        corr = self.model(tnf_batch)
        score_neg = self.normalised_max_correlation(corr)

        # loss
        loss = score_neg - score_pos
        return loss

    def forward(self, tnf_batch):
        corr = self.model(tnf_batch)
        return self.weak_supervision(corr, tnf_batch)


class SparseStrongWeakLoss(WeakLoss):
    def __init__(self, image_size, model, N=64, im_fe_ratio=16, fine_coarse_ratio = 4, loss_name='meanfnorm', backbone='',
                 weight_orthogonal=0.1, alpha=0.1, weight_loss=[0.0, 1, 0.],
                 orth_samples=3, gauss_size=5, mode=1):
        super(SparseStrongWeakLoss, self).__init__(image_size, model, loss_name=loss_name,
                                                   weight_orthogonal=weight_orthogonal)

        self.inv_interp = interpolator.InverInterpolator(int(im_fe_ratio / fine_coarse_ratio), kernel_size=gauss_size,
                                                         mode=mode, N=N)

        self.weight_loss = weight_loss
        self.extract_featuremap = tools.ExtractFeatureMap(im_fe_ratio)
        self.backbone = backbone
        self.im_fe_ratio = im_fe_ratio

        if 'orthogonal' in loss_name:
            self.orth = True

            # weak orthogonal
            start, end = (self.image_size - 1) * (1 / 4), (self.image_size - 1) * (orth_samples / 4)
            x, y = np.meshgrid(np.linspace(start, end, orth_samples), np.linspace(start, end, orth_samples))
            x, y = torch.FloatTensor(x).view(1, -1, 1), torch.FloatTensor(y).view(1, -1, 1)
            self.sampled_keypoints = torch.cat((x, y), 2)  # 1 x 9 x 2
            self.sampled_keypoints = self.sampled_keypoints.cuda()

            self.identity = torch.eye(orth_samples * orth_samples).cuda()
        else:
            self.orth = False

        # self.celoss = nn.CrossEntropyLoss()

    def set_weight(self, weight_loss):
        self.weight_loss = weight_loss

    def orth_core(self, X, Orth_gt):
        """
        Return loss [tensor] B
        """
        B = X.shape[0]
        Xt = X.permute(0, 2, 1)
        Xo = torch.bmm(X, Xt)

        Dif = (Xo - Orth_gt)  # .diagonal(dim1=1, dim2=2) # Trace ( Xo - Xg ) #????  B x N x N
        Dif = Dif.view(B, -1)
        loss = torch.norm(Dif, dim=1)
        return loss

    def strong_orthogonal(self, X, G, key_num_gt):
        Gt = G.permute(0, 2, 1)
        Xog = torch.bmm(G, Gt)

        loss = self.orth_core(X, Xog)
        loss = torch.div(loss, key_num_gt)
        return loss.mean()

    def weak_orthogonal(self, X, I):
        B, N, _ = X.shape
        loss = self.orth_core(X, I) / N
        return loss.mean()

    def diff(self, keycorr, onehot, key_num_gt):
        return ((keycorr - onehot).norm(dim=2).sum(dim=1) / key_num_gt).mean();

    def forward(self, tnf_batch):
        """
        Arguments:
            tnf_batch [dict] source_image, target_image, source_points, target_points, assignment
        Return:
            loss
        """
        # sparse loss
        corr, feature_A0, feature_B0 = self.model(tnf_batch)
        _, _, HA, WA = feature_A0.shape
        _, _, HB, WB = feature_B0.shape
        ratio = np.round(feature_A0.shape[2] / corr.shape[2])
        ratio = int(ratio)

        loss = torch.zeros(1, 1).cuda()
        o_loss = torch.zeros(1, 1).cuda()
        reg_loss = torch.zeros(1, 1).cuda()

        if self.weight_loss[0] > 0:  # correlation regularisation term
            # corr [B,1,H1,W1,H2,W2]
            reg_loss += self.extract_featuremap.regularise_corr(corr, source_to_target=True)
            reg_loss += self.extract_featuremap.regularise_corr(corr, source_to_target=False)
            reg_loss *= self.weight_loss[0]
            loss += reg_loss

        if self.weight_loss[1] > 0:  # strong loss
            Xg = tnf_batch["assignment"]
            Xgt = tnf_batch["assignment"].permute(0, 2, 1)

            src_gt = tnf_batch["source_points"]  # B x 32 x 3
            dst_gt = tnf_batch["target_points"]  # B x 32 x 3
            B, N, _ = src_gt.shape

            # calc key_num
            src_indices_gt, src_key_num_gt = tools.calc_gt_indices(src_gt, Xg)
            dst_indices_gt, dst_key_num_gt = tools.calc_gt_indices(dst_gt, Xgt)

            # firstly, do from source to target
            keycorrB_A = self.extract_featuremap(corr, src_gt[:, :, :2], source_to_target=True)
            keycorrB_A_Lv0 = tools.sparse_feature_correlation(src_gt[:, :, :2], ratio, feature_A0, feature_B0)
            keycorrB_A = tools.mask_over_corr(keycorrB_A, keycorrB_A_Lv0)
            # then from target to source
            keycorrA_B = self.extract_featuremap(corr, dst_gt[:, :, :2], source_to_target=False)
            keycorrA_B_Lv0 = tools.sparse_feature_correlation(dst_gt[:, :, :2], ratio, feature_B0, feature_A0)
            keycorrA_B = tools.mask_over_corr(keycorrA_B, keycorrA_B_Lv0)
            keycorrB_A = keycorrB_A.view(B, N, -1)
            keycorrA_B = keycorrA_B.view(B, N, -1)
            onehotB_A = self.inv_interp(Xg, dst_gt[:, :, :2], HB, WB)  # B x N x H1W1
            onehotA_B = self.inv_interp(Xgt, src_gt[:, :, :2], HA, WA)


            # MSELoss
            if self.inv_interp.mode <= 2:
                strong_loss = self.diff(keycorrB_A, onehotB_A, src_key_num_gt)
                strong_loss += self.diff(keycorrA_B, onehotA_B, dst_key_num_gt)
            elif self.inv_interp.mode == 3:  # using multiple ground truth onehot which are produces using various gaussian blur level.
                strong_loss = self.diff(keycorrB_A, onehotB_A[0], src_key_num_gt)
                strong_loss += self.diff(keycorrB_A, onehotB_A[2], src_key_num_gt)
                strong_loss += self.diff(keycorrA_B, onehotA_B[0], dst_key_num_gt)
                strong_loss += self.diff(keycorrA_B, onehotA_B[2], dst_key_num_gt)

            strong_loss *= self.weight_loss[1]

            if self.orth:
                if self.inv_interp.mode <= 2:
                    so_loss = self.strong_orthogonal(keycorrB_A, onehotB_A, src_key_num_gt) + self.strong_orthogonal(
                        keycorrA_B, onehotA_B, dst_key_num_gt)
                if self.inv_interp.mode == 3:
                    so_loss = self.strong_orthogonal(keycorrB_A, onehotB_A[0], src_key_num_gt) + self.strong_orthogonal(
                        keycorrA_B, onehotA_B[0], dst_key_num_gt)
                    so_loss += self.strong_orthogonal(keycorrB_A, onehotB_A[1],
                                                      src_key_num_gt) + self.strong_orthogonal(keycorrA_B, onehotA_B[1],
                                                                                               dst_key_num_gt)

                loss += (1 - self.weight_orthogonal) * strong_loss + self.weight_orthogonal * so_loss
                o_loss += so_loss
            else:
                loss += strong_loss

        else:
            strong_loss = torch.zeros((1, 1)).cuda()

        if self.weight_loss[2] > 0:
            # positive
            score_pos = self.normalised_max_correlation(corr)
            # negative
            b = tnf_batch['source_image'].size(0)
            v = np.roll(np.arange(b), -1)
            tnf_batch['source_image'] = tnf_batch['source_image'][v, :]  # roll
            corr, _, _ = self.model(tnf_batch)

            score_neg = self.normalised_max_correlation(corr)
            weak_loss = score_neg - score_pos
            weak_loss *= self.weight_loss[2]
            loss += weak_loss

        else:
            weak_loss = torch.zeros((1, 1)).cuda()

        return loss, strong_loss, weak_loss, o_loss, reg_loss