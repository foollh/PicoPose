
import os
import sys
import json
import cv2
import trimesh
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
import imgaug.augmenters as iaa
from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa

from utils.data_utils import (
    load_im,
    io_load_gt,
    io_load_masks,
    get_point_cloud_from_depth,
    get_bbox,
)
from utils.template_utils import R_opencv2R_opengl, get_obj_poses_from_template_level


class Dataset():
    def __init__(self, cfg, num_img_per_epoch=-1):
        self.cfg = cfg

        self.data_dir = cfg.data_dir
        self.num_img_per_epoch = num_img_per_epoch
        self.min_visib_px = cfg.min_px_count_visib
        self.min_visib_frac = cfg.min_visib_fract
        self.dilate_mask = cfg.dilate_mask
        self.rgb_mask_flag = cfg.rgb_mask_flag
        self.size_ratio = cfg.size_ratio
        self.img_size = cfg.img_size
        self.augment_real = cfg.augment_real
        self.augment_tem = cfg.augment_tem

        self.data_paths = [
            os.path.join('MegaPose-GSO', 'train_pbr_web'),
            os.path.join('MegaPose-ShapeNetCore', 'train_pbr_web')
        ]
        self.templates_paths = [
            os.path.join(self.data_dir, 'MegaPose-Templates', 'GSO'),
            os.path.join(self.data_dir, 'MegaPose-Templates', 'ShapeNetCore'),
        ]
        self.templates_K = np.array(
            [572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]
        ).reshape((3, 3))
        self.avail_index, self.template_poses = get_obj_poses_from_template_level(
            'utils/predefined_poses', 
            1,
            'all',
            return_cam=False,
            return_index=True,
        )
        # we use the location to find for nearest template on the sphere
        template_openGL_poses = R_opencv2R_opengl(self.template_poses[:, :3, :3])
        self.obj_template_openGL_locations = template_openGL_poses[:, 2, :3]  # Nx3

        self.dataset_paths = []
        for f in self.data_paths:
            with open(os.path.join(self.data_dir, f, 'key_to_shard.json')) as fr:
                key_shards = json.load(fr)

                for k in key_shards.keys():
                    path_name = os.path.join(f, "shard-" + f"{key_shards[k]:06d}", k)
                    self.dataset_paths.append(path_name)

        self.length = len(self.dataset_paths)
        print('Total {} images .....'.format(self.length))


        with open(os.path.join(self.data_dir, self.data_paths[0], 'gso_models.json')) as fr:
            self.model_info = [json.load(fr)]
        with open(os.path.join(self.data_dir, self.data_paths[1], 'shapenet_models.json')) as fr:
            self.model_info.append(json.load(fr))

        # gdrnpp aug 
        aug_code = (
            "Sequential(["
            "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
            "Sometimes(0.4, GaussianBlur((0., 3.))),"
            "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
            "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
            "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
            "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
            "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
            "Sometimes(0.3, Invert(0.2, per_channel=True)),"
            "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
            "Sometimes(0.5, Multiply((0.6, 1.4))),"
            "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
            "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
            "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"
            "], random_order=True)"
            # cosy+aae
        )
        self.color_augmentor = eval(aug_code)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                std=[0.26862954, 0.26130258, 0.27577711])])


    def __len__(self):
        return self.length if self.num_img_per_epoch == -1 else self.num_img_per_epoch

    def reset(self):
        if self.num_img_per_epoch == -1:
            self.num_img_per_epoch = self.length

        num_img = self.length
        if num_img <= self.num_img_per_epoch:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch, replace=False)


    def __getitem__(self, index):
        while True:  # return valid data for train
            processed_data = self.read_data(self.img_idx[index])
            if processed_data is None:
                index = self._rand_another(index)
                continue
            return processed_data

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def read_data(self, index):
        path_head = self.dataset_paths[index]
        dataset_type = path_head.split('/')[0][9:]
        if not self._check_path(os.path.join(self.data_dir, path_head)):
            return None

        real_data = self.process_real(path_head, )
        if real_data is None:
            return None

        # template
        view_id = self.sample_template(real_data['pose'][:3, :3])   
        template_data = self.process_template(dataset_type, real_data['obj_id'], view_id)
        if template_data is None:
            return None

        ret_dict = {
            # real_data
            'real_full_depth': torch.FloatTensor(real_data['full_depth']),
            'real_rgb': torch.FloatTensor(real_data['rgb']),
            'real_bbox': torch.FloatTensor(real_data['bbox']), 
            'real_mask': torch.FloatTensor(real_data['mask']), 
            'real_M': torch.FloatTensor(real_data['M']), 
            'real_K': torch.FloatTensor(real_data['K']),
            'real_pose': torch.FloatTensor(real_data['pose']),
            # template data
            'tem_full_depth': torch.FloatTensor(template_data['full_depth']),
            'tem_rgb': torch.FloatTensor(template_data['rgb']),
            'tem_bbox': torch.FloatTensor(template_data['bbox']), 
            'tem_mask': torch.FloatTensor(template_data['mask']),
            'tem_M': torch.FloatTensor(template_data['M']), 
            'tem_K': torch.FloatTensor(template_data['K']),
            'tem_pose': torch.FloatTensor(template_data['pose']),
        }
        return ret_dict

    def process_real(self, path_head):
        # gt_info
        gt_info = io_load_gt(open(os.path.join(self.data_dir, path_head+'.gt_info.json'), 'rb'))
        valid_idx = []
        for k, item in enumerate(gt_info):
            if item['px_count_valid'] >= self.min_visib_px and item['visib_fract'] >= self.min_visib_frac:
                valid_idx.append(k)
        if len(valid_idx) == 0:
            return None
        num_instance = len(valid_idx)
        valid_idx = valid_idx[np.random.randint(0, num_instance)]
        gt_info = gt_info[valid_idx]

        # gt
        gt = io_load_gt(open(os.path.join(self.data_dir, path_head+'.gt.json'), 'rb'))[valid_idx]
        obj_id = gt['obj_id']
        pose = np.eye(4)
        pose[:3, :3] = np.array(gt['cam_R_m2c']).reshape(3,3).astype(np.float32)
        pose[:3, 3] = np.array(gt['cam_t_m2c']).reshape(3).astype(np.float32) / 1000.0
        
        # camera
        camera = json.load(open(os.path.join(self.data_dir, path_head+'.camera.json'), 'rb'))
        K = np.array(camera['cam_K']).reshape(3,3)

        # mask
        mask = io_load_masks(open(os.path.join(self.data_dir, path_head+'.mask_visib.json'), 'rb'))[valid_idx]
        if np.sum(mask) == 0:
            return None
        if self.dilate_mask and np.random.rand() < 0.5:
            mask = np.array(mask>0).astype(np.uint8)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=4)

        size_ratio = random.uniform(*(1.0, self.size_ratio))
        bbox = get_bbox(mask>0, size_ratio)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]
        if len(choose) < 32:
            return None

        # rgb+mask resize
        image = load_im(os.path.join(self.data_dir, path_head+'.rgb.jpg')).astype(np.uint8)
        rgb = image[..., ::-1][y1:y2, x1:x2, :]
        if self.augment_real and np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb.copy(), (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[:,:,None].astype(int), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        rgb = self.transform(np.array(rgb / 255)).float()

        # depth
        depth = load_im(os.path.join(self.data_dir, path_head+'.depth.png')).astype(np.float32)
        depth = depth * camera['depth_scale'] / 1000.0
        
        M_crop = np.array([
            [1, 0, -bbox[2]], 
            [0, 1, -bbox[0]], 
            [0, 0, 1]],dtype=np.float32)
        M_resize = np.array([
            [self.img_size / (y2-y1), 0, 0], 
            [0, self.img_size / (x2-x1), 0], 
            [0, 0, 1]],dtype=np.float32)
        M = M_resize @ M_crop

        
        return {
            'full_depth': depth, 
            'rgb': rgb, 
            'mask': mask, 
            'bbox': bbox, 
            'M': M, 
            'K': K, 
            'pose': pose, 
            'obj_id': obj_id
        }

    def process_template(self, type, obj_id, view_id):
        if type == 'GSO':
            info = self.model_info[0][obj_id]
            assert info['obj_id'] == obj_id
            template_dir = os.path.join(
                self.templates_paths[0],
            )

        elif type == 'ShapeNetCore':
            info = self.model_info[1][obj_id]
            assert info['obj_id'] == obj_id
            template_dir = os.path.join(
                self.templates_paths[1],
            )

        image_path = f"{template_dir}/{obj_id:06d}/{view_id:06d}.png"
        depth_path = f"{template_dir}/{obj_id:06d}/{view_id:06d}_depth.png"
        # assert os.path.exists(image_path), f"{image_path} does not exist"
        if not os.path.exists(image_path):
            return None
        
        if not os.path.exists(depth_path):
            depth_path = depth_path.replace("_blenderproc", "")
            assert os.path.exists(depth_path), f"{depth_path} does not exist"
        rgba  = load_im(image_path)
        rgb = rgba[..., :3]
        mask = (rgba[..., 3] / 255).astype(np.float32)
        if np.sum(mask) == 0:
            return None
        size_ratio = random.uniform(*(1.0, self.size_ratio))
        bbox = get_bbox(mask>0, size_ratio)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = rgb.astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]
        if self.augment_tem and np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[:,:,None].astype(int), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        rgb = self.transform(np.array(rgb / 255)).float()
        
        # depth
        depth = load_im(depth_path) * 0.1 / 1000.0

        # pose
        object_pose = np.load(os.path.join(template_dir, 'object_poses', f'{obj_id:06d}'+'.npy'))[view_id]
        object_pose[:3, 3] = object_pose[:3, 3] * 0.1 / 1000.0

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
            'full_depth': depth,
            'rgb': rgb, 
            'mask': mask, 
            'bbox': bbox, 
            'M': M,
            'K': self.templates_K,
            'pose': object_pose,
        }

    def sample_template(self, object_rot, topk=5):
        # convert query pose to OpenGL coordinate
        query_opencv_query_R = object_rot
        query_opengl_R = R_opencv2R_opengl(query_opencv_query_R)
        query_opengl_location = query_opengl_R[2, :3]  # Mx3

        # find the nearest template
        distances = np.linalg.norm(
            query_opengl_location - self.obj_template_openGL_locations, axis=1
        )
        view_ids = np.argsort(distances)[:topk]
        view_id = np.random.choice(view_ids)
        return view_id

    def _check_path(self, path_head):
        keys = [
            '.camera.json',
            '.depth.png',
            '.gt_info.json',
            '.gt.json',
            '.mask_visib.json',
            '.rgb.jpg'
        ]

        for k in keys:
            if not os.path.exists(path_head + k):
                return False
        return True

