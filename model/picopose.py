import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.dinov2.dinov2 import DINOV2
from model.regressor import Regressor
from model.dpt import DPTHead
from model.flow_decoder import FlowDecoder
from utils.correspondence import compute_init_correspondences, compute_stage3_correspondences
from utils.loss_utils import compute_stage_one_loss, compute_stage_two_loss, compute_stage_three_loss
from utils.matching import matching_templates, matching_features_similarity
from utils.pose_recovery import pose_recovery_2d_prediction
from utils.torch_utils import calc_pred_Ms, apply_affine
from utils.augment import aug_gtM_noise
from utils.keypoints import KeypointInput, KeyPointSampler


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.keypoint_sampler = KeyPointSampler()

        self.backbone = DINOV2(cfg.stage1)
        self.regressor = Regressor(cfg.stage2)
        self.dpt = DPTHead(cfg.stage3)
        self.decoder = FlowDecoder(cfg.stage3)


    def compute_keypoint_data(self, end_points):
        rel_pose = end_points['tem_pose'] @ (end_points['real_pose'].inverse())
        T_real2template, T_template2real = rel_pose, rel_pose.inverse()
        tar_data = KeypointInput(
            full_depth=end_points['real_full_depth'],
            K=end_points['real_K'],
            M=end_points['real_M'],
            mask=end_points['real_mask'],
        )
        src_data = KeypointInput(
            full_depth=end_points['tem_full_depth'],
            K=end_points['tem_K'],
            M=end_points['tem_M'],
            mask=end_points['tem_mask'],
        )
        keypoint_data = self.keypoint_sampler.sample_pts(
            tar_data=tar_data,
            src_data=src_data,
            T_src2target=T_template2real,
            T_tar2source=T_real2template,
        )
        return keypoint_data


    def select_template_data(self, end_points, pred_id_src, k):
        selected_end_points = {}
        # template
        selected_end_points['tem_pose'] = torch.gather(end_points['tem_pose'], 1, pred_id_src[:, k][:,None,None,None].repeat(1,1,4,4)).squeeze(1)
        selected_end_points['tem_K'] = torch.gather(end_points['tem_K'], 1, pred_id_src[:, k][:,None,None,None].repeat(1,1,3,3)).squeeze(1)
        selected_end_points['tem_M'] = torch.gather(end_points['tem_M'], 1, pred_id_src[:, k][:,None,None,None].repeat(1,1,3,3)).squeeze(1)
        hm, wm = end_points['tem_mask'].shape[2:]
        selected_end_points['tem_mask'] = torch.gather(end_points['tem_mask'], 1, pred_id_src[:, k][:,None,None,None].repeat(1,1,hm,wm)).squeeze(1)
        selected_end_points['tem_rgb'] = torch.gather(end_points['tem_rgb'], 1, pred_id_src[:, k][:,None,None,None,None].repeat(1,1,3,hm,wm)).squeeze(1)
        hp, wp = end_points['tem_pts3d'].shape[2:4]
        selected_end_points['tem_pts3d'] = torch.gather(end_points['tem_pts3d'], 1, pred_id_src[:, k][:,None,None,None,None].repeat(1,1,hp,wp,3)).squeeze(1)

        # real
        selected_end_points['real_pts2d'] = end_points['real_pts2d']
        selected_end_points['real_K'] = end_points['real_K']
        selected_end_points['real_M'] = end_points['real_M']
        selected_end_points['real_mask'] = end_points['real_mask']
        selected_end_points['real_pose'] = end_points['real_pose']
        return selected_end_points

    def forward_test_hyp(self, end_points, features_real):
        output = {}
        output['tem_pose'] = end_points['tem_pose']
        output['tar_pts_2d'], output['src_pts_3d'] = end_points['real_pts2d'].permute(0,3,2,1), end_points['tem_pts3d'].permute(0,3,1,2)
        
        ################################################ stage 1 ################################################
        features_tem = self.backbone(end_points['tem_rgb'])

        ################################################ stage 2 ################################################
        sim = matching_features_similarity(features_tem[-1], features_real[-1], end_points['tem_mask'], end_points['real_mask'])  
        pred_translation, pred_scale, pred_inplane = self.regressor(sim)

        pred_Ms = calc_pred_Ms(pred_scale, pred_inplane, pred_translation, end_points['tem_pose'], end_points['tem_K'], end_points['tem_M'])
        # recovery from template pose
        output['pred_poses'] = pose_recovery_2d_prediction(
            end_points['real_M'], end_points['real_K'], pred_Ms, 
            end_points['tem_K'], end_points['tem_M'], end_points['tem_pose']
        )
        ################################################ stage 3 ################################################
        init_flow, init_certainty = compute_init_correspondences(pred_Ms, end_points['tem_mask'])
        features_tem, features_real = self.dpt(features_tem), self.dpt(features_real)
        pred_flow, pred_certainty = self.decoder(features_tem, features_real, init_flow, init_certainty)
        output['pred_tar_pts'], output['pred_src_pts'] = compute_stage3_correspondences(pred_flow[-1], pred_certainty[-1], threshold=0.5)

        return output

    def forward_test(self, end_points, hyp=5):
        features_real = self.backbone(end_points['real_rgb'])
        feature_tem = F.normalize(end_points['template_feature'], dim=2)

        # matching templates
        pred_score_src, pred_id_src = matching_templates(
            feature_tem, features_real[-1], end_points['tem_mask'], end_points['real_mask'], topk=hyp
        )
        
        outputs = []
        for k in range(hyp):
            selected_end_points = self.select_template_data(end_points, pred_id_src, k)
            output = self.forward_test_hyp(selected_end_points, features_real)
            outputs.append(output)

        return outputs

    def forward_train(self, end_points):
        # compute gt correspondences
        keypoint_data = self.compute_keypoint_data(end_points)

        ################################################ stage 1 ################################################
        features_real = self.backbone(end_points['real_rgb'])
        features_tem = self.backbone(end_points['tem_rgb'])
        # loss
        end_points['loss_info'] = compute_stage_one_loss(features_tem[-1], features_real[-1], keypoint_data['src_pts'], keypoint_data['tar_pts']) 

        ################################################ stage 2 ################################################
        sim = matching_features_similarity(features_tem[-1], features_real[-1], end_points['tem_mask'], end_points['real_mask']) 
        pred_translation, pred_scale, pred_inplane = self.regressor(sim)
        # loss
        end_points['loss_2d_trans'], end_points['loss_scale'], end_points['loss_inplane'] = compute_stage_two_loss(end_points, pred_translation, pred_scale, pred_inplane)
        
        ################################################ stage 3 ################################################
        pred_Ms = aug_gtM_noise(end_points)
        init_flow, init_certainty = compute_init_correspondences(pred_Ms, end_points['tem_mask'])
        features_tem, features_real = self.dpt(features_tem), self.dpt(features_real)
        pred_flow, pred_certainty = self.decoder(features_tem, features_real, init_flow, init_certainty)
        # loss
        end_points = compute_stage_three_loss(end_points, pred_flow, pred_certainty, keypoint_data['tar_pts'])

        return end_points
        
    def forward(self, end_points, hyp=5):
        if self.training:
            return self.forward_train(end_points)
        else:
            return self.forward_test(end_points, hyp)
