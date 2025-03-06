import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.corr_lookup import CorrLookup, bilinear_sample, coords_grid
from model.stage3.raft_decoder import MotionEncoder, XHead, CorrelationPyramid


class FlowDecoder(nn.Module):
    def __init__(self, num_levels, radius) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

        self.proj = []
        self.corr_block, self.corr_lookup = [], []
        self.encoder = []
        self.flow_pred, self.mask_pred = [], []
        for lvl in range(self.num_levels):
            self.proj.append(
                nn.Sequential(nn.Conv2d(256, 256, 1, 1), 
                nn.BatchNorm2d(256))
            )
            radius = int(self.radius / 2)
            self.corr_block.append(CorrelationPyramid(num_levels=lvl+1))
            self.corr_lookup.append(CorrLookup(radius=radius))

            net_type = 'Basic'
            conv_cfg, norm_cfg, act_cfg = None, None, dict(type='ReLU')
            self.encoder.append(MotionEncoder(
                num_levels=lvl+1,
                radius=radius,
                net_type=net_type,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            )

            self.flow_pred.append(XHead(2*256+128, [512, 256], 2, x='flow'))
            self.mask_pred.append(XHead(2*256+128, [512, 256], 1, x='mask'))
            
        self.proj = nn.ModuleList(self.proj)
        self.corr_block = nn.ModuleList(self.corr_block)
        self.corr_lookup = nn.ModuleList(self.corr_lookup)
        self.encoder = nn.ModuleList(self.encoder)
        self.flow_pred = nn.ModuleList(self.flow_pred)
        self.mask_pred = nn.ModuleList(self.mask_pred)

    def feature_sample(self, feature, flow, mode='bilinear', padding_mode='zeros', align_corners=True):
        B, _, H, W = flow.shape
        xx = torch.arange(0, W, device=flow.device)
        yy = torch.arange(0, H, device=flow.device)
        grid = coords_grid(B, xx, yy) + flow  # shape N, 2, H, W
        grid = grid.permute(0, 2, 3, 1)  # shape N, H, W, 2
        feature_hat = bilinear_sample(feature, grid, mode, padding_mode, align_corners)
        return feature_hat

    def forward_flow(self, feat_render, feat_real, flow, level):
        corr_pyramid = self.corr_block[level](feat_render, feat_real)
        
        corr = self.corr_lookup[level](corr_pyramid, flow)
        motion_feat = self.encoder[level](corr, flow)

        feat_real_hat = self.feature_sample(feat_real, flow)

        x = torch.cat([feat_render, feat_real_hat, motion_feat], dim=1)
        # predict delta flow
        delta_flow_pred = self.flow_pred[level](x)
        # predict mask
        delta_certainty_pred = self.mask_pred[level](x)
        
        return delta_flow_pred, delta_certainty_pred

    def forward(self, feat_render_list, feat_real_list, init_flow, init_certainty, iters=1):
        pred_flow, pred_certainty = [], []
        flow, certainty = init_flow, init_certainty
        for level in range(self.num_levels):
            feat_render, feat_real = self.proj[level](feat_render_list[level]), self.proj[level](feat_real_list[level])
            
            for i in range(iters):
                delta_flow, delta_certainty = self.forward_flow(feat_render, feat_real, flow, level)
                flow = flow + delta_flow
                certainty = certainty + delta_certainty
            
            pred_flow.append(flow)
            pred_certainty.append(certainty)

            if level != self.num_levels-1:
                flow = 2 * F.interpolate(
                    flow, scale_factor=(2, 2), mode='bilinear', align_corners=True)
                certainty = F.interpolate(
                    certainty, scale_factor=(2, 2), mode='bilinear', align_corners=True)
            
        return pred_flow, pred_certainty

