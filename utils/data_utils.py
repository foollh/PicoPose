import os
import numpy as np
import json
import imageio
import cv2

from PIL import Image

def load_im(path, tool="imageio"):
    """Loads an image from a file.

    :param path: Path to the image file to load.
    :return: ndarray with the loaded image.
    """
    if tool == "imageio":
        im = imageio.imread(path)
    elif tool == "pil":
        im = Image.open(path)
    else:
        raise TypeError
    return im


def io_load_gt(
    gt_file,
    instance_ids=None,
):
    """Load ground truth from an I/O object.
    Instance_ids can be specified to load only a
    subset of object instances.

    :param gt_file: I/O object that can be read with json.load.
    :param instance_ids: List of instance ids.
    :return: List of ground truth annotations (one dict per object instance).
    """
    gt = json.load(gt_file)
    if instance_ids is not None:
        gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
    gt = [_gt_as_numpy(gt_n) for gt_n in gt]
    return gt


def io_load_masks(
    mask_file,
    instance_ids=None
):
    """Load object masks from an I/O object.
    Instance_ids can be specified to apply RLE
    decoding to a subset of object instances contained
    in the file.

    :param mask_file: I/O object that can be read with json.load.
    :param masks_path: Path to json file.
    :return: a [N,H,W] binary array containing object masks.
    """
    masks_rle = json.load(mask_file)
    masks_rle = {int(k): v for k, v in masks_rle.items()}
    if instance_ids is None:
        instance_ids = masks_rle.keys()
        instance_ids = sorted(instance_ids)
    masks = np.stack([
        rle_to_binary_mask(masks_rle[instance_id])
        for instance_id in instance_ids])
    return masks


def _gt_as_numpy(gt):
    if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = \
        np.array(gt['cam_R_m2c'], np.float64).reshape((3, 3))
    if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = \
        np.array(gt['cam_t_m2c'], np.float64).reshape((3, 1))
    return gt


def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


def get_point_cloud_from_depth(depth, K, bbox=None):
    cam_fx, cam_fy, cam_cx, cam_cy = K[0,0], K[1,1], K[0,2], K[1,2]

    im_H, im_W = depth.shape
    xmap = np.array([[i for i in range(im_W)] for j in range(im_H)])
    ymap = np.array([[j for i in range(im_W)] for j in range(im_H)])

    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
        xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
        ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

    pt2 = depth.astype(np.float32)
    pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
    pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

    cloud = np.stack([pt0,pt1,pt2]).transpose((1,2,0))
    return cloud


def get_resize_rgb_choose(choose, bbox, img_size):
    rmin, rmax, cmin, cmax = bbox
    crop_h = rmax - rmin
    ratio_h = img_size / crop_h
    crop_w = cmax - cmin
    ratio_w = img_size / crop_w

    row_idx = choose // crop_h
    col_idx = choose % crop_h
    choose = (np.floor(row_idx * ratio_w) * img_size + np.floor(col_idx * ratio_h)).astype(np.int64)
    return choose


def get_bbox(label, size_ratio=1.0):
    img_width, img_length = label.shape
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    b = min(max(r_b, c_b), min(img_width, img_length)) * size_ratio
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

    rmin = center[0] - int(b / 2)
    rmax = center[0] + int(b / 2)
    cmin = center[1] - int(b / 2)
    cmax = center[1] + int(b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return [rmin, rmax, cmin, cmax]

def get_square_bbox(bbox, img_size, size_ratio=1.0):
    img_width, img_length = img_size
    rmin, rmax, cmin, cmax = bbox
    r_b = rmax - rmin
    c_b = cmax - cmin
    b = min(max(r_b, c_b), min(img_width, img_length)) * size_ratio
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

    rmin = center[0] - int(b / 2)
    rmax = center[0] + int(b / 2)
    cmin = center[1] - int(b / 2)
    cmax = center[1] + int(b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return [rmin, rmax, cmin, cmax]

def get_random_rotation():
    angles = np.random.rand(3) * 2 * np.pi
    rand_rotation = np.array([
        [1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]),0,np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]),-np.sin(angles[2]),0],
        [np.sin(angles[2]), np.cos(angles[2]),0],
        [0,0,1]
    ])
    return rand_rotation

def get_model_info(obj, return_color=False, sample_num=2048):
    if return_color:
        model_points, model_color, symmetry_flag = obj.get_item(return_color, sample_num)
        return (model_points, model_color, symmetry_flag)
    else:
        model_points,  symmetry_flag = obj.get_item()
        return (model_points, symmetry_flag)

def get_bop_depth_map(inst):
    scene_id, img_id, data_folder = inst['scene_id'], inst['img_id'], inst['data_folder']
    try:
        depth = np.array(Image.open(os.path.join(data_folder, f'{scene_id:06d}', 'depth', f'{img_id:06d}.png'))) / 1000.0
    except:
        depth = np.array(Image.open(os.path.join(data_folder, f'{scene_id:06d}', 'depth', f'{img_id:06d}.tif'))) / 1000.0
    return depth

def get_bop_image(inst, bbox, img_size, mask=None, mask_flag=True):
    scene_id, img_id, data_folder = inst['scene_id'], inst['img_id'], inst['data_folder']
    rmin, rmax, cmin, cmax = bbox
    img_path = os.path.join(data_folder, f'{scene_id:06d}/')

    strs = [f'rgb/{img_id:06d}.jpg', f'rgb/{img_id:06d}.png', f'gray/{img_id:06d}.tif']
    for s in strs:
        if os.path.exists(os.path.join(img_path,s)):
            img_path = os.path.join(img_path,s)
            break

    rgb = load_im(img_path).astype(np.uint8) 
    if len(rgb.shape)==2:
        rgb = np.concatenate([rgb[:,:,None], rgb[:,:,None], rgb[:,:,None]], axis=2)
    rgb = rgb[..., ::-1][rmin:rmax, cmin:cmax, :3] / 255.0
    if mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask[:,:,None].astype(int), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return rgb, mask

def get_bop_mask(inst, gt_idx, bbox, img_size):
    scene_id, img_id, data_folder = inst['scene_id'], inst['img_id'], inst['data_folder']
    mask_path = os.path.join(data_folder, f'{scene_id:06d}', 'mask_visib', f'{img_id:06d}_{gt_idx:06d}.png')
    
    rmin, rmax, cmin, cmax = bbox
    mask = load_im(mask_path).astype(np.uint8) / 255.0
    mask = mask[rmin:rmax, cmin:cmax]
    mask = cv2.resize(mask[:,:,None].astype(int), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask

def get_rgb_path(inst):
    scene_id, img_id, data_folder = inst['scene_id'], inst['img_id'], inst['data_folder']
    img_path = os.path.join(data_folder, f'{scene_id:06d}/')

    strs = [f'rgb/{img_id:06d}.jpg', f'rgb/{img_id:06d}.png', f'gray/{img_id:06d}.tif']
    for s in strs:
        if os.path.exists(os.path.join(img_path,s)):
            img_path = os.path.join(img_path,s)
            break
    return img_path

def crop_transform(rgba, bbox, target_size):
    scale = target_size / max(bbox[2]-bbox[0], bbox[3]-bbox[1])
    M_crop, M_resize_pad = np.eye(3), np.eye(3)

    # crop and scale
    rgba = rgba[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    M_crop[:2, 2] = -np.array(bbox[:2])

    rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    M_resize_pad[:2, :2] *= scale

    if rgba.shape[0] / rgba.shape[1] != 1:
        pad_top = (target_size - rgba.shape[0]) // 2
        pad_bottom = target_size - rgba.shape[0] - pad_top
        pad_bottom = max(pad_bottom, 0)

        pad_left = (target_size - rgba.shape[1]) // 2
        pad_left = max(pad_left, 0)
        pad_right = target_size - rgba.shape[1] - pad_left

        rgba = cv2.copyMakeBorder(rgba, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        M_resize_pad[:2, 2] = np.array([pad_left, pad_top])

    M = M_resize_pad @ M_crop

    # sometimes, 1 pixel is missing due to rounding, so interpolate again
    rgba = cv2.resize(rgba, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return rgba, M