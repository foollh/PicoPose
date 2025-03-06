import os
import sys
import os.path as osp
import time
import logging
import argparse
import numpy as np
import random
import importlib
import pickle as cPickle
from tqdm import tqdm
import json
import torch
from torch import optim
from omegaconf import OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'model'))


from utils.lr_scheduler import WarmupCosineLR
from utils.lite import Lite
from utils.loss_utils  import Loss
from utils.pose_recovery import pose_recovery_ransac_pnp

# cnos
detetion_paths = {
    "itodd": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json",
    "hb": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json",
    "icbin": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json",
    "lmo": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
    "tless": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json",
    "ycbv": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json",
    "tudl": "data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json",
}

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="index of gpu")
    parser.add_argument("--model",
                        type=str,
                        default="picopose",
                        help="name of model")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="tudl",
                        help="")
    parser.add_argument("--checkpoint_path",
                        type=str, 
                        default="none", 
                        help="path to checkpoint file")
    parser.add_argument("--iter",
                        type=int,
                        default=400000,
                        help="iter num. for testing")
    parser.add_argument("--view",
                        type=int,
                        default=-1,
                        help="view number of templates")
    parser.add_argument("--version_id",
                        type=int,
                        default=0,
                        help="experiment id")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    log_dir = osp.join("log", args.model, 'version_'+str(args.version_id))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = OmegaConf.load(args.config)
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.checkpoint_path = args.checkpoint_path
    cfg.test_iter = args.iter
    cfg.dataset = args.dataset

    if args.view != -1:
        cfg.test_dataset.n_template_view = args.view

    assert len(cfg.gpus) == 1
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus

    return cfg



def run_test(model, cfg, save_path, dataset_name, detetion_path):
    model.eval()
    bs = cfg.test_dataloader.bs

    # build dataloader
    dataset = importlib.import_module(cfg.test_dataset.name)
    dataset = dataset.BOPTestset(cfg.test_dataset, dataset_name, detetion_path)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=cfg.test_dataloader.num_workers,
            shuffle=cfg.test_dataloader.shuffle,
            sampler=None,
            drop_last=cfg.test_dataloader.drop_last,
            pin_memory=cfg.test_dataloader.pin_memory
        )

    # prepare for templates data
    templates_data = dataset.get_templates('cuda')
    template_features = []
    for obj_id, obj_idx in tqdm(dataset.obj_idxs.items()):
        template_features.append([])
        with torch.no_grad():
            n_batch = int(np.ceil(dataset.n_template_view/bs))
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = dataset.n_template_view if j == n_batch-1 else (j+1) * bs
                features = model.backbone(templates_data['tem_rgb'][obj_idx][start_idx:end_idx].contiguous())
                template_features[obj_idx].append(features[-1])
            template_features[obj_idx] = torch.cat(template_features[obj_idx]).squeeze()

    templates_data['template_feature'] = torch.stack(template_features)
    

    hyp = cfg.model.hypothesis
    total_time = 0
    lines = []
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            torch.cuda.synchronize()
            for key in data:
                data[key] = data[key].cuda()
            n_instance = data['score'].size(1)
            n_batch = int(np.ceil(n_instance/bs))

            preds_image = []

            end = time.time()
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = n_instance if j == n_batch-1 else (j+1) * bs
                obj_idx = data['obj_idx'][0][start_idx:end_idx].reshape(-1)

                # process inputs
                inputs = {}
                for key in data:
                    inputs[key] = data[key][0][start_idx:end_idx].contiguous()

                for key in templates_data:
                    inputs[key] = templates_data[key][obj_idx].contiguous()

                # make predictions
                with torch.no_grad():
                    outputs = model(inputs, hyp)
                
                for k in range(end_idx-start_idx):
                    preds = []
                    for tk in range(hyp):
                        pred = {}

                        pred_r_3, pred_t_3, inliers_ratio, success = pose_recovery_ransac_pnp(
                            outputs[tk]['tar_pts_2d'][k], outputs[tk]['src_pts_3d'][k], inputs["real_K"][k], 
                            outputs[tk]['tem_pose'][k], outputs[tk]['pred_tar_pts'][k], outputs[tk]['pred_src_pts'][k]
                        )
                        if not success:
                            pred_r_3, pred_t_3 = outputs[tk]['pred_poses'][k][:3, :3], outputs[tk]['pred_poses'][k][:3, 3]
                            pred_r_3, pred_t_3 = pred_r_3.detach().cpu().numpy(), pred_t_3.detach().cpu().numpy()
                        
                        pred['R_stage_3'] = pred_r_3.reshape(9)
                        pred['t_stage_3'] = pred_t_3.reshape(3) * 1000
                        pred['inliers_ratio'] = inliers_ratio
                        preds.append(pred)
                    
                    preds_image.append(sorted(preds, key=lambda x: x["inliers_ratio"], reverse=True))

            image_time = time.time() - end

            # write results
            scene_id = data['scene_id'].item()
            img_id = data['img_id'].item()
            image_time += data['seg_time'].item()
            for k in range(n_instance):
                pred_score = data['score'][0][k].item()

                # csv
                line = ','.join((
                    str(scene_id),
                    str(img_id),
                    str(data['obj_id'][0][k].item()),
                    str(pred_score),
                    ' '.join((str(v) for v in preds_image[k][0]['R_stage_3'])),
                    ' '.join((str(v) for v in preds_image[k][0]['t_stage_3'])),
                    f'{image_time}\n',
                ))
                lines.append(line)

            total_time += image_time

            t.set_description(
                "Test [{}/{}]".format(i+1, len(dataloder))
            )
            t.update(1)

    print(total_time/(i+1))

    save_path = os.path.join(save_path, 'picopose-stage3-'+ str(hyp)+'hyp_' + dataset_name +'-test.csv')
    with open(save_path, 'w+') as f:
        f.writelines(lines)
    print('saving to {} ...'.format(save_path))



if __name__ == "__main__":
    cfg = init()

    print("************************ Start Logging ************************")
    print(cfg)
    print("using gpu: {}".format(cfg.gpus))

    # model
    print("creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    if cfg.checkpoint_path == 'none':
        files = [f for f in os.listdir(os.path.join(cfg.log_dir, 'checkpoints')) if str(cfg.test_iter).zfill(6) in f]
        assert len(files) == 1
        checkpoint = os.path.join(cfg.log_dir, 'checkpoints', files[0])
    else:
        checkpoint = cfg.checkpoint_path
    # optimizer
    learning_rate = cfg.optimizer.lr
    betas = cfg.optimizer.betas
    eps = cfg.optimizer.eps
    weight_decay = cfg.optimizer.weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

    # learning rate
    if cfg.trainer.strategy == 'ddp':
        lr_max_iters = int(cfg.lr_scheduler.max_iters / len(cfg.trainer.devices))
    else:
        lr_max_iters = cfg.lr_scheduler.max_iters
    warmup_factor, warmup_iters = cfg.lr_scheduler.warmup_factor, cfg.lr_scheduler.warmup_iters
    lr_scheduler = WarmupCosineLR(optimizer, max_iters=lr_max_iters, warmup_factor=warmup_factor, warmup_iters=warmup_iters, )
    lr_scheduler_config = {
        "scheduler": lr_scheduler,
        "interval": "step",
        "frequency": 1,
    }
    test_lite = Lite(
        network=model,
        loss=Loss(),
        optimizer=optimizer,
        lr_scheduler_config=lr_scheduler_config,
        dataloaders=None
    )
    test_model = test_lite.load_from_checkpoint(checkpoint, network=model, loss=Loss(), optimizer=optimizer, lr_scheduler_config=lr_scheduler_config, dataloaders=None)

    if cfg.dataset == 'all':
        datasets = ['ycbv', 'tudl',  'lmo', 'icbin', 'tless', 'itodd' , 'hb']
        for dataset_name in datasets:
            print('begining evaluation on {} ...'.format(dataset_name))

            save_path = os.path.join(cfg.log_dir, dataset_name + '_eval_iter' + str(cfg.test_iter).zfill(6))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            run_test(test_model.network, cfg, save_path, dataset_name, detetion_paths[dataset_name])

            print('saving to {} ...'.format(save_path))
            print('finishing evaluation on {} ...'.format(dataset_name))

    else:
        dataset_name = cfg.dataset
        print('begining evaluation on {} ...'.format(dataset_name))

        save_path = os.path.join(cfg.log_dir, dataset_name + '_eval_iter' + str(cfg.test_iter).zfill(6))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        run_test(test_model.network, cfg,  save_path, dataset_name, detetion_paths[dataset_name])

        print('finishing evaluation on {} ...'.format(dataset_name))


