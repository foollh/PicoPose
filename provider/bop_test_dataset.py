
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as cocomask
import copy
import torch
import torchvision.transforms as transforms

from utils.data_utils import (
    get_model_info,
    get_bop_depth_map,
    get_bop_image, 
    get_bop_mask, 
    get_rgb_path, 
    get_bbox,
    load_im, 
    get_point_cloud_from_depth, 
    get_square_bbox, 
)
from utils.bop_object_utils import load_objs
from utils.torch_utils import init_points2d_numpy

class BOPTestset():
    def __init__(self, cfg, eval_dataset_name='ycbv', detetion_path=None):
        assert detetion_path is not None

        self.cfg = cfg
        self.dataset = eval_dataset_name
        self.data_dir = cfg.data_dir
        self.rgb_mask_flag = cfg.rgb_mask_flag
        self.img_size = cfg.img_size
        self.minimum_n_point = cfg.minimum_n_point
        self.seg_filter_score = cfg.seg_filter_score
        self.n_template_view = cfg.n_template_view
        self.pts_size = cfg.pts_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                std=[0.26862954, 0.26130258, 0.27577711])])

        if eval_dataset_name == 'tless':
            model_path = 'models_cad'
        else:
            model_path = 'models'
        self.template_folder = os.path.join(cfg.template_dir, eval_dataset_name)

        self.data_folder = os.path.join(self.data_dir, eval_dataset_name, 'test')
        self.model_folder = os.path.join(self.data_dir, eval_dataset_name, model_path)
        obj, obj_ids = load_objs(self.model_folder, )
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.objects = obj
        self.obj_idxs = obj_idxs

        self.templates_K = np.array(
            [572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]
        ).reshape((3, 3))

        with open(detetion_path) as f:
            dets = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation

        self.det_keys = []
        self.dets = {}
        for det in tqdm(dets, 'processing detection results'):
            scene_id = det['scene_id'] # data type: int
            img_id = det['image_id'] # data type: int

            key = str(scene_id).zfill(6) + '_' + str(img_id).zfill(6)
            if key not in self.det_keys:
                self.det_keys.append(key)
                self.dets[key] = []
            self.dets[key].append(det)
        del dets
        print('testing on {} images on {}...'.format(len(self.det_keys), eval_dataset_name))

        target_dets_path = os.path.join(self.data_dir, eval_dataset_name, 'test_targets_bop19.json')
        with open(target_dets_path) as f:
            target_dets = json.load(f)
        
        det_hyp = 1
        self.best_dets = {det_key:[] for det_key in self.det_keys}
        for idx, target in enumerate(target_dets):
            target_obj_id = target['obj_id']
            scene_id, img_id = target["scene_id"], target["im_id"]
            image_key = f"{scene_id:06d}_{img_id:06d}"

            dets_per_image = self.dets[image_key]
            dets = [det for det in dets_per_image if (det["category_id"] == target_obj_id)]
            if len(dets) == 0:  # done in MegaPose
                dets = copy.deepcopy(dets_per_image)
                for det in dets:
                    det["category_id"] = target_obj_id
            assert len(dets) > 0

            # sort the detections by score descending
            dets = sorted(
                dets,
                key=lambda x: x["score"],
                reverse=True,
            )
            # keep only the top detections
            num_instances = target["inst_count"] * det_hyp
            dets = dets[:num_instances]
            for det_ in dets:
                self.best_dets[image_key].append(det_)

    def __len__(self):
        return len(self.det_keys)

    def __getitem__(self, index):
        dets = self.best_dets[self.det_keys[index]]

        instances = []
        for det in dets:
            if det['score']>self.seg_filter_score:
                real_data = self.get_instance(det)
                if real_data is None:
                    continue

                instance = {
                    'score': torch.FloatTensor([real_data['score']]),
                    'obj_id': torch.IntTensor([real_data['obj_id']]), 
                    'obj_idx': torch.IntTensor([real_data['obj_idx']]), 
                    # real_data
                    'real_pts2d': torch.FloatTensor(real_data['pts2d']), 
                    'real_rgb': torch.FloatTensor(real_data['rgb']),
                    'real_bbox': torch.FloatTensor(real_data['bbox']), 
                    'real_mask': torch.FloatTensor(real_data['mask']), 
                    'real_M': torch.FloatTensor(real_data['M']), 
                    'real_K': torch.FloatTensor(real_data['K']),
                    'real_pose': torch.FloatTensor(real_data['pose']),
                }
                if instance is not None:
                    instances.append(instance)

        ret_dict = {}
        for key in instances[0].keys():
            ret_dict[key] = torch.stack([instance[key] for instance in instances])
        ret_dict['scene_id'] = torch.IntTensor([int(self.det_keys[index][0:6])])
        ret_dict['img_id'] = torch.IntTensor([int(self.det_keys[index][7:13])])
        ret_dict['seg_time'] = torch.FloatTensor([dets[0]['time']])
        return ret_dict

    def get_instance(self, data):
        scene_id = data['scene_id'] # data type: int
        img_id = data['image_id'] # data type: int
        obj_id = data['category_id'] # data type: int
        bbox = data['bbox'] # list, len:4
        seg = data['segmentation'] # keys: counts, size
        score = data['score']

        scene_folder = os.path.join(self.data_folder, f'{scene_id:06d}')
        scene_camera = json.load(open(os.path.join(scene_folder, 'scene_camera.json')))
        K = np.array(scene_camera[str(img_id)]['cam_K']).reshape((3, 3)).copy()
        depth_scale = scene_camera[str(img_id)]['depth_scale']
        inst = dict(scene_id=scene_id, img_id=img_id, data_folder=self.data_folder)

        obj_idx = self.obj_idxs[obj_id]
        pose = np.eye(4)

        # depth
        depth = get_bop_depth_map(inst) * depth_scale

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, depth > 0)
        if np.sum(mask) > self.minimum_n_point:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            y1, y2, x1, x2 = get_square_bbox([bbox[1], bbox[1]+bbox[3], bbox[0], bbox[0]+bbox[2]], (h,w))
        
        mask = mask[y1:y2, x1:x2]
        # choose = mask.astype(np.float32).flatten().nonzero()[0]
        # if np.sum(mask) < self.minimum_n_point:
        #     return None

        # pts
        pts = get_point_cloud_from_depth(depth, self.templates_K, [y1, y2, x1, x2])
        pts = cv2.resize(pts, (self.pts_size, self.pts_size), interpolation=cv2.INTER_NEAREST)

        # rgb
        rgb, mask = get_bop_image(inst, [y1,y2,x1,x2], self.img_size, mask, self.rgb_mask_flag)
        rgb = self.transform(np.array(rgb)).float()

        M_crop = np.array([
            [1, 0, -bbox[2]], 
            [0, 1, -bbox[0]], 
            [0, 0, 1]],dtype=np.float32)
        M_resize = np.array([
            [self.img_size / (y2-y1), 0, 0], 
            [0, self.img_size / (x2-x1), 0], 
            [0, 0, 1]],dtype=np.float32)
        M = M_resize @ M_crop

        # pts2d
        one_vector = np.ones((self.pts_size, self.pts_size, 1))
        pts2d = init_points2d_numpy(self.img_size, patch_size=self.img_size/self.pts_size)
        pts2d = np.concatenate((pts2d, one_vector), axis=2)
        pts2d = np.linalg.inv(M) @ pts2d.reshape(-1,3).transpose(1,0)
        pts2d = (pts2d[:2] / pts2d[2:]).transpose(1,0).reshape(self.pts_size,self.pts_size,2)

        return {
            'pts2d':pts2d, 
            'rgb': rgb, 
            'pts': pts, 
            'mask': mask, 
            'bbox': bbox, 
            'M': M, 
            'K': K, 
            'pose': pose, 
            'obj_id': obj_id,
            'obj_idx': obj_idx, 
            'score': score
        }


    def _get_template(self, obj_id, view_id):
        image_path = f"{self.template_folder}/{obj_id:06d}/{view_id:06d}.png"
        depth_path = f"{self.template_folder}/{obj_id:06d}/{view_id:06d}_depth.png"
        # assert os.path.exists(image_path), f"{image_path} does not exist"
        if not os.path.exists(image_path):
            return None
        
        if not os.path.exists(depth_path):
            depth_path = depth_path.replace("_blenderproc", "")
            assert os.path.exists(depth_path), f"{depth_path} does not exist"
        rgba  = load_im(image_path)
        rgb = rgba[..., :3]
        mask = (rgba[..., 3] / 255).astype(np.float32)
        bbox = get_bbox(mask)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]

        # depth
        depth = load_im(depth_path) / 1000.0
        # pts
        pts = get_point_cloud_from_depth(depth, self.templates_K, [y1, y2, x1, x2])
        pts = cv2.resize(pts, (self.pts_size, self.pts_size), interpolation=cv2.INTER_NEAREST)

        rgb = rgb[..., ::-1][y1:y2, x1:x2, :] / 255.0
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[:,:,None].astype(int), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        rgb = self.transform(np.array(rgb)).float()

        # pose
        object_pose = np.load(os.path.join(self.template_folder, 'object_poses', f'{obj_id:06d}'+'.npy'))[view_id]
        object_pose[:3, 3] = object_pose[:3, 3] / 1000.0

        M_crop = np.array([
            [1, 0, -bbox[2]], 
            [0, 1, -bbox[0]], 
            [0, 0, 1]],dtype=np.float32)
        M_resize = np.array([
            [self.img_size / (y2-y1), 0, 0], 
            [0, self.img_size / (x2-x1), 0], 
            [0, 0, 1],], dtype=np.float32)
        M = M_resize @ M_crop

        return {
            'rgb': rgb, 
            'pts3d': pts, 
            'mask': mask, 
            'bbox': bbox, 
            'M': M,
            'K': self.templates_K,
            'pose': object_pose,
        }

    def get_templates(self, device):
        n_object = len(self.objects)
        n_template_view = self.n_template_view
        all_tem_rgb = [[] for i in range(n_object)]
        all_tem_pts3d = [[] for i in range(n_object)]
        all_tem_mask = [[] for i in range(n_object)]
        all_tem_bbox = [[] for i in range(n_object)]
        all_tem_M = [[] for i in range(n_object)]
        all_tem_K = [[] for i in range(n_object)]
        all_tem_pose = [[] for i in range(n_object)]
        
        for i in range(n_template_view):
            for obj_id, obj_idx in self.obj_idxs.items():
                template_data = self._get_template(obj_id, i)
                
                all_tem_rgb[obj_idx].append(torch.FloatTensor(template_data['rgb']))
                all_tem_pts3d[obj_idx].append(torch.FloatTensor(template_data['pts3d']))
                all_tem_mask[obj_idx].append(torch.FloatTensor(template_data['mask']))
                all_tem_bbox[obj_idx].append(torch.FloatTensor(template_data['bbox']))
                all_tem_M[obj_idx].append(torch.FloatTensor(template_data['M']))
                all_tem_K[obj_idx].append(torch.FloatTensor(template_data['K']))
                all_tem_pose[obj_idx].append(torch.FloatTensor(template_data['pose']))

        for i in range(n_object):
            all_tem_rgb[i] = torch.stack(all_tem_rgb[i]).to(device)
            all_tem_pts3d[i] = torch.stack(all_tem_pts3d[i]).to(device)
            all_tem_mask[i] = torch.stack(all_tem_mask[i]).to(device)
            all_tem_bbox[i] = torch.stack(all_tem_bbox[i]).to(device)
            all_tem_M[i] = torch.stack(all_tem_M[i]).to(device)
            all_tem_K[i] = torch.stack(all_tem_K[i]).to(device)
            all_tem_pose[i] = torch.stack(all_tem_pose[i]).to(device)

        templates_data = {
            'tem_rgb': torch.stack(all_tem_rgb),
            'tem_pts3d': torch.stack(all_tem_pts3d),
            'tem_bbox': torch.stack(all_tem_bbox), 
            'tem_mask': torch.stack(all_tem_mask),
            'tem_M': torch.stack(all_tem_M), 
            'tem_K': torch.stack(all_tem_K),
            'tem_pose': torch.stack(all_tem_pose),
        }

        return templates_data



