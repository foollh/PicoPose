import torch
import torch.nn as nn

from .dpt import DPTHead
from .flow_decoder import FlowDecoder



class OffsetRegressor(nn.Module):
    def __init__(self, cfg, ):
        super().__init__()

        self.dpt_head = DPTHead(cfg.nclass, cfg.in_channels, features=256, use_bn=True, out_channels=[256, 512, 1024, 1024], use_clstoken=False)
        self.flow_decoder = FlowDecoder(cfg.num_levels, cfg.radius)

    def forward(self, features_tem, features_real, init_flow, init_certainty):
        features_tem, features_real = self.dpt_head(features_tem), self.dpt_head(features_real)
        pred_flow, pred_certainty = self.flow_decoder(features_tem, features_real, init_flow, init_certainty) 
        return pred_flow, pred_certainty

