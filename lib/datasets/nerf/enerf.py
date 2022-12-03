import numpy as np
import os
from glob import glob
from lib.utils.data_utils import load_K_Rt_from_P, read_cam_file
from lib.config import cfg
import imageio
import tqdm
from multiprocessing import Pool
import copy
import cv2
import random
from lib.config import cfg
from lib.utils import data_utils
from PIL import Image
import torch
import json
from lib.datasets import enerf_utils

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas()

    def build_metas(self):
        if len(self.scenes) == 0:
            scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        pairs = torch.load('data/mvsnerf/pairs.th')
        for scene in scenes:
            json_info = json.load(open(os.path.join(self.data_root, scene,'transforms_train.json')))
            b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            scene_info = {'ixts': [], 'exts': [], 'img_paths': []}
            for idx in range(len(json_info['frames'])):
                c2w = np.array(json_info['frames'][idx]['transform_matrix'])
                c2w = c2w @ b2c
                ext = np.linalg.inv(c2w)
                ixt = np.eye(3)
                ixt[0][2], ixt[1][2] = 400., 400.
                focal = .5 * 800 / np.tan(.5 * json_info['camera_angle_x'])
                ixt[0][0], ixt[1][1] = focal, focal
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                img_path = os.path.join(self.data_root, scene, 'train/r_{}.png'.format(idx))
                scene_info['img_paths'].append(img_path)
            self.scene_infos[scene] = scene_info
            train_ids, render_ids = pairs[f'{scene}_train'], pairs[f'{scene}_val']
            if self.split == 'train':
                render_ids = train_ids
            c2ws = np.stack([np.linalg.inv(scene_info['exts'][idx]) for idx in train_ids])
            for idx in render_ids:
                c2w = np.linalg.inv(scene_info['exts'][idx])
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)

                argsorts = distance.argsort()
                argsorts = argsorts[1:] if idx in train_ids else argsorts

                input_views_num = cfg.enerf.train_input_views[1] + 1 if self.split == 'train' else cfg.enerf.test_input_views
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, idx, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        scene_info['scene_name'] = scene
        tar_img, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        ret = {'src_inps': src_inps.transpose(0, 3, 1, 2),
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        tar_mask = np.ones_like(tar_img[..., 0]).astype(np.uint8)
        H, W = tar_img.shape[:2]
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})
        near_far = np.array([2.5, 5.5]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.enerf.cas_config.num):
            rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            s = cfg.enerf.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H*s), f'w_{i}': int(W*s)})
        return ret

    def read_src(self, scene, src_views):
        src_ids = src_views
        ixts, exts, imgs = [], [], []
        for idx in src_ids:
            img = self.read_image(scene, idx)
            imgs.append((img*2-1).astype(np.float32))
            ixt, ext = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img = self.read_image(scene, view_idx)
        ixt, ext = self.read_cam(scene, view_idx)
        return img, ext, ixt

    def read_cam(self, scene, view_idx):
        ext = scene['exts'][view_idx]
        ixt = scene['ixts'][view_idx]
        return ixt, ext

    def read_image(self, scene, view_idx):
        img_path = scene['img_paths'][view_idx]
        img = (np.array(imageio.imread(img_path)) / 255.).astype(np.float32)
        img = (img[..., :3] * img[..., -1:] + (1 - img[..., -1:])).astype(np.float32)
        return img

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

