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
from lib.datasets import enerf_utils

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas()

    def build_metas(self):
        if len(self.scenes) == 0:
            scenes = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        pairs = torch.load('data/mvsnerf/pairs.th')
        for scene in scenes:

            pose_bounds = np.load(os.path.join(self.data_root, scene, 'poses_bounds.npy')) # c2w, -u, r, -t
            poses = pose_bounds[:, :15].reshape((-1, 3, 5))
            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
            ixts = np.eye(3)[None].repeat(len(poses), 0)
            ixts[:, 0, 0], ixts[:, 1, 1] = poses[:, 2, 4], poses[:, 2, 4]
            ixts[:, 0, 2], ixts[:, 1, 2] = poses[:, 1, 4]/2., poses[:, 0, 4]/2.
            ixts[:, :2] *= 0.25

            img_paths = sorted([item for item in os.listdir(os.path.join(self.data_root, scene, 'images_4')) if '.png' in item])
            depth_ranges = pose_bounds[:, -2:]
            scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': img_paths, 'depth_ranges': depth_ranges.astype(np.float32)}
            scene_info['scene_name'] = scene
            self.scene_infos[scene] = scene_info

            train_ids = pairs[f'{scene}_train']
            if self.split == 'train':
                render_ids = train_ids
            else:
                render_ids = pairs[f'{scene}_val']

            c2ws = c2ws[train_ids]
            for i in render_ids:
                c2w = scene_info['c2ws'][i]
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if i in train_ids else argsorts
                if self.split == 'train':
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.train_input_views[1]+1]]
                else:
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.test_input_views]]
                self.metas += [(scene, i, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        ret = {'src_inps': src_inps.transpose(0, 3, 1, 2),
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})

        H, W = tar_img.shape[:2]
        depth_ranges = np.array(scene_info['depth_ranges'])
        near_far = np.array([depth_ranges[:, 0].min().item(), depth_ranges[:, 1].max().item()]).astype(np.float32)
        # near_far = scene_info['depth_ranges'][tar_view]
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view}})

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
            img, orig_size = self.read_image(scene, idx)
            imgs.append(((img/255.)*2-1).astype(np.float32))
            ixt, ext, _ = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img, orig_size = self.read_image(scene, view_idx)
        img = (img/255.).astype(np.float32)
        ixt, ext, _ = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        return ixt, np.linalg.inv(ext), 1

    def read_image(self, scene, view_idx):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'images_4', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        orig_size = img.shape[:2][::-1]
        img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        return np.array(img), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

