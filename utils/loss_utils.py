
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from .torch_utils import cosSin, calc_gt_trans_scale_inplane, get_relative_outplane, gather
from .corr_lookup import coords_grid


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, end_points):
        out_dicts = {'loss': 0}
        for key in end_points.keys():
            if 'loss' in key:
                out_dicts[key] = end_points[key].mean()
                out_dicts['loss'] = out_dicts['loss'] + end_points[key]
        out_dicts['loss'] = torch.clamp(out_dicts['loss'], max=100.0).mean()
        return out_dicts


class RAFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, max_flow=400, eps=1e-10):
        super().__init__()
        self.loss_weight = loss_weight
        self.max_flow = max_flow
        self.eps = eps
    
    def forward(self, pred_flow:torch.Tensor, gt_flow:torch.Tensor, valid=None):
        mag = torch.sum(gt_flow**2, dim=1).sqrt()
        if valid is None:
            valid = (mag < self.max_flow).to(gt_flow)
        else:
            valid = ((valid >= 0.5) & (mag < self.max_flow)).to(gt_flow)
        loss = (pred_flow - gt_flow).abs()
        loss = (valid[:, None] * loss).sum() / (valid.sum() + self.eps)
        return self.loss_weight * loss


class TranslationLoss(nn.Module):
    def __init__(self, loss="l2", log=False):
        super(TranslationLoss, self).__init__()
        self.loss = loss
        self.log = log

    def forward(self, pred_scale, gt_scale):
        """
        pred_scale: (B,)
        gt_scale: (B,)
        """
        if self.log:
            pred_scale = torch.log(pred_scale.clamp(min=1e-6))
            gt_scale = torch.log(gt_scale)
        if self.loss == "l1":
            loss = F.l1_loss(pred_scale, gt_scale)
        else:
            loss = F.mse_loss(pred_scale, gt_scale)
        assert not torch.isnan(loss)
        return loss


class ScaleLoss(nn.Module):
    def __init__(self, loss="l2", log=False):
        super(ScaleLoss, self).__init__()
        self.loss = loss
        self.log = log

    def forward(self, pred_scale, gt_scale):
        """
        pred_scale: (B,)
        gt_scale: (B,)
        """
        if self.log:
            pred_scale = torch.log(pred_scale.clamp(min=5e-3))
            gt_scale = torch.log(gt_scale)
        if self.loss == "l1":
            loss = F.l1_loss(pred_scale, gt_scale)
        else:
            loss = F.mse_loss(pred_scale, gt_scale)
        assert not torch.isnan(loss)
        return loss


class InplaneLoss(nn.Module):
    def __init__(self, loss="l2", normalize=False):
        super(InplaneLoss, self).__init__()
        self.normalize = normalize
        self.loss = loss
        self.eps = 1e-6

    def forward(self, pred_cos_sin, gt_cos_sin):
        """
        pred_inp_R: (B, 2)
        gt_inp_R: (B, 2)
        """
        if self.normalize:
            pred_cos_sin = F.normalize(pred_cos_sin, dim=1)
            gt_cos_sin = F.normalize(gt_cos_sin, dim=1)
        if self.loss == "geodesic":
            pred_cos = pred_cos_sin[:, 0]
            pred_sin = pred_cos_sin[:, 1]
            gt_cos = gt_cos_sin[:, 0]
            gt_sin = gt_cos_sin[:, 1]
            cos_diff = pred_cos * gt_cos + pred_sin * gt_sin
            cos_diff = torch.clamp(cos_diff, -1 + self.eps, 1 - self.eps)
            loss = torch.acos(cos_diff).mean()
        elif self.loss == "l1":
            loss = F.l1_loss(pred_cos_sin, gt_cos_sin)
        elif self.loss == "l2":
            loss = F.mse_loss(pred_cos_sin, gt_cos_sin)
        else:
            raise NotImplementedError
        assert not torch.isnan(loss)
        return loss



def compute_flow_loss(pred_flow, pred_mask, gt_flow, gt_mask, mask_weight=10, flow_weight=0.1):
    # mask
    mask_loss = mask_weight * F.binary_cross_entropy_with_logits(pred_mask[:, 0], gt_mask.float())
    # flow
    raft_loss = RAFTLoss(loss_weight=flow_weight)
    flow_loss = raft_loss(pred_flow, gt_flow.permute(0,3,1,2), gt_mask)
    return flow_loss, mask_loss

def compute_outplane_loss(pred_outplane, gt_outplane, outplane_weight=1):
    outplane_loss = InplaneLoss(loss="geodesic", normalize=False)
    loss_x = outplane_loss(pred_outplane[0], cosSin(gt_outplane[0])) 
    loss_y = outplane_loss(pred_outplane[1], cosSin(gt_outplane[1])) 
    return (loss_x + loss_y)*outplane_weight

def compute_scale_inplane_loss(pred_scale, gt_scale, pred_inplane, gt_inplane, scale_log=False, scale_weight=1, inplane_weight=1):
    scale_loss = ScaleLoss(loss="l2", log=scale_log)
    inplane_loss = InplaneLoss(loss="geodesic", normalize=False)
    return scale_loss(pred_scale, gt_scale) * scale_weight, inplane_loss(pred_inplane, cosSin(gt_inplane)) * inplane_weight 

def compute_2d_translation_loss(pred_translation, gt_translation, trans_weight=1, loss_type='l2'):
    translation_loss = TranslationLoss(loss=loss_type)
    return translation_loss(pred_translation, gt_translation) * trans_weight


def compute_stage_one_loss(src_feat, tar_feat, src_pts, tar_pts, tau=0.1):
    b, _, h, w = src_feat.shape
    hs = ws = int(src_pts.shape[1]**0.5)

    src_pts_ = src_pts.reshape(b,hs,ws,2)
    src_mask = (src_pts_[..., 0] == -1).float()
    src_mask = F.interpolate(src_mask[:,None], size=(h,w), mode='nearest').squeeze(1).bool()
    src_pts_ = (h/hs) * F.interpolate(src_pts_.permute(0,3,1,2), size=(h,w), mode='nearest').permute(0,2,3,1)
    src_pts_[src_mask] = -1
    src_pts_ = src_pts_.reshape(b, -1, 2)

    tar_pts_ = tar_pts.reshape(b,hs,ws,2)
    tar_mask = (tar_pts_[..., 0] == -1).float()
    tar_mask = F.interpolate(tar_mask[:,None], size=(h,w), mode='nearest').squeeze(1).bool()
    tar_pts_ = (h/hs) * F.interpolate(tar_pts_.permute(0,3,1,2), size=(h,w), mode='nearest').permute(0,2,3,1)
    tar_pts_[tar_mask] = -1
    tar_pts_ = tar_pts_.reshape(b, -1, 2)

    src_feat_ = gather(src_feat, src_pts_.long())
    tar_feat_ = gather(tar_feat, tar_pts_.long())
    labels = torch.arange(src_feat_.shape[0], dtype=torch.long).to(src_feat_.device)

    # compute infonce loss
    query_feat = F.normalize(src_feat_, dim=1)
    ref_feats = F.normalize(tar_feat_, dim=1)
    logits = query_feat @ ref_feats.t()
    logits = logits / tau
    loss = F.cross_entropy(logits, labels)
    return loss

def compute_stage_two_loss(end_points, pred_translation, pred_scale, pred_inplane, trans_scale=14):
    # calculate gt 2d translation, scale, inplane
    gt_2d_translation, gt_relScale, gt_relInplane = calc_gt_trans_scale_inplane(end_points)

    # calculate 2d translation loss 
    loss_2d_translation = compute_2d_translation_loss(pred_translation, gt_2d_translation / trans_scale, trans_weight=1, loss_type='l1')
    loss_scale, loss_inplane = compute_scale_inplane_loss(pred_scale, gt_relScale, pred_inplane, gt_relInplane, scale_log=True, scale_weight=1, inplane_weight=1)

    return loss_2d_translation, loss_scale, loss_inplane

def compute_stage_three_loss(end_points, pred_flow, pred_certainty, tar_pts):
    B = tar_pts.shape[0]
    Hs = Ws = int(tar_pts.shape[1]**0.5)
    tar_pts = rearrange(tar_pts, 'b (h w) c -> b w h c', h=Hs)
    tar_mask = torch.logical_and(tar_pts[..., 1] != -1, tar_pts[..., 0] != -1).float()

    for idx, (flow, certainty) in enumerate(zip(pred_flow, pred_certainty)):
        B, _, H, W = flow.shape
        xx = torch.arange(0, W, device=flow.device)
        yy = torch.arange(0, H, device=flow.device)
        grid = coords_grid(B, xx, yy).permute(0, 2, 3, 1)

        gt_certainty = F.interpolate(tar_mask[:,None], size=(H,W), mode='nearest').squeeze(1).bool()
        gt_flow = (H/Hs) * F.interpolate(tar_pts.permute(0,3,1,2), size=(H,W), mode='nearest').permute(0,2,3,1)
        gt_flow = gt_flow * gt_certainty[...,None] - grid

        end_points['loss_flow'+str(idx)], end_points['loss_certainty'+str(idx)] = compute_flow_loss(flow, certainty, gt_flow, gt_certainty, 1, 0.1)

    return end_points