import numpy as np
import os
from glob import glob
from lib.utils.data_utils import load_K_Rt_from_P, read_cam_file
from lib.datasets import enerf_utils
from lib.config import cfg
import imageio
from tqdm import tqdm
from multiprocessing import Pool
import copy
import cv2
from os.path import join
import random
from lib.config import cfg
from lib.utils import data_utils
from lib.utils import base_utils
import trimesh
import torch
import matplotlib.pyplot as plt
if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_ratio = kwargs['input_ratio']
        self.input_h_w = kwargs['input_h_w']
        self.frames = kwargs['frames']
        self.scene = kwargs['scene']
        self.build_metas(kwargs)

    def build_metas(self, config):
        self.metas = []
        scene_info = {'ixts': [], 'exts': [], 'bbox': {}}
        scene_root = join(self.data_root, self.scene)
        self.scene_root = scene_root
        intri_path = join(scene_root, 'intri.yml')
        extri_path = join(scene_root, 'extri.yml')
        cams = data_utils.read_camera(intri_path, extri_path)
        self.cams = cams
        cam_ids = sorted([item for item in os.listdir(join(scene_root, 'images')) if item[0] != '.']) # filter .DS_STORE
        cam_len = len(cam_ids)
        ixts =  np.array([self.cams[cam_id]['K'] for cam_id in cam_ids]).reshape(cam_len, 3, 3).astype(np.float32)
        exts =  np.array([self.cams[cam_id]['RT'] for cam_id in cam_ids]).reshape(cam_len, 3, 4).astype(np.float32)
        Ds =  np.array([self.cams[cam_id]['dist'] for cam_id in cam_ids]).reshape(cam_len, 5).astype(np.float32)
        exts_ones = np.zeros_like(exts[:, :1, :])
        exts_ones[..., 3] = 1.
        exts = np.concatenate([exts, exts_ones], axis=1)
        c2ws = np.linalg.inv(exts)
        scene_info['exts'] = exts
        scene_info['ixts'] = ixts
        scene_info['Ds'] = Ds

        frame_len = len(glob(f'{scene_root}'))
        b, e, s = config['frames']
        e = e if e != -1 else frame_len
        self.scene_info = scene_info

        for frame_id in tqdm(np.arange(frame_len)[b:e:s]):
            bounds = np.load(join(scene_root, 'vhull', '{:06d}.npy'.format(frame_id)))
            corners_3d = base_utils.get_bound_corners(bounds)
            scene_info['bbox'][frame_id] = corners_3d

        points = np.array(trimesh.load(os.path.join(scene_root, 'background.ply')).vertices)
        scene_info['bbox_dynamic'] = points
        self.bkgd_near_far = []
        for view_id in range(len(cam_ids)):
            img, ext, ixt = self.read_data(view_id, 0)
            h, w = img.shape[:2]
            points_ = points @ ext[:3, :3].T + ext[:3, 3].T
            uv = points_ @ ixt.T
            uv[:, :2] = uv[:, :2] / uv[:, 2:]
            mask = np.logical_or(uv[..., 0] < 0, uv[..., 1] < 0)
            mask = np.logical_or(mask, uv[..., 0] > w - 1)
            mask = np.logical_or(mask, uv[..., 1] > h - 1)
            uv = uv[mask == False]
            near_far = np.array([uv[:, 2].min(), uv[:, 2].max()])
            self.bkgd_near_far.append(near_far)

        input_views = config['input_views']
        b, e, s = input_views
        e = e if e != -1 else cam_len
        input_views = np.arange(cam_len)[b:e:s]
        render_views = config['render_views']
        b, e, s = render_views
        e = e if e != -1 else cam_len
        render_views = np.arange(cam_len)[b:e:s]

        train_cam_pos = c2ws[:, :3, 3][input_views]

        b, e, s = config['frames']
        e = e if e != -1 else frame_len
        for tar_view in render_views:
            cam_pos = c2ws[:, :3, 3][tar_view]
            distance = np.linalg.norm(train_cam_pos - cam_pos[None], axis=-1)
            argsorts = distance.argsort()
            input_views_num = cfg.enerf.train_input_views[-1] + 1 if self.split == 'train' else cfg.enerf.test_input_views
            if tar_view in input_views:
                src_views = [input_views[i] for i in argsorts[:input_views_num]]
            else:
                src_views = [input_views[i] for i in argsorts[1:input_views_num+1]]
            self.metas += [(tar_view, src_views, frame_id) for frame_id in np.arange(frame_len)[b:e:s]]


    def read_data(self, view_id, frame_id):
        img_path = join(self.scene_root, 'images', '{:02d}'.format(view_id), '{:06d}.jpg'.format(frame_id))
        img = np.array(imageio.imread(img_path) / 255.).astype(np.float32)

        ext = np.array(self.scene_info['exts'][view_id])
        ixt = np.array(self.scene_info['ixts'][view_id]).copy()
        D = np.array(self.scene_info['Ds'][view_id]).copy()

        img = cv2.undistort(img, ixt, D)
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            ixt[:2] *= self.input_ratio

        if self.input_h_w is not None:
            H, W, _ = img.shape
            h, w = self.input_h_w
            crop_h = int((H - h) * 0.65) #crop more
            crop_h_ = (H - h) - crop_h
            crop_w = int((W - w) * 0.5)
            crop_w_ = W - w - crop_w
            img = img[crop_h:-crop_h_, crop_w:-crop_w_]
            ixt[1, 2] -= crop_h
            ixt[0, 2] -= crop_w
        return img, ext, ixt

    def read_tar(self, view_id, frame_id):
        img, ext, ixt = self.read_data(view_id, frame_id)
        corners_3d = self.scene_info['bbox'][frame_id] @ ext[:3, :3].T + ext[:3, 3].T
        bound_mask = data_utils.get_bound_2d_mask(corners_3d, ixt, img.shape[0], img.shape[1])
        near_far = np.array([corners_3d[:, 2].min(), corners_3d[:, 2].max()])
        x, y, w, h = cv2.boundingRect(bound_mask.astype(np.uint8))
        w_ori, h_ori = w, h
        w = (w//32 + 1) * 32 if w%32 != 0 or w == 0 else w
        h = (h//32 + 1) * 32 if h%32 != 0 or h == 0 else h
        x -= (w - w_ori) // 2
        y -= (h - h_ori) // 2
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x = img.shape[1] - w if (x + w) > img.shape[1] else x
        y = img.shape[0] - h if (y + h) > img.shape[0] else y
        return img, ext, ixt, np.array([[x,y,w,h]]).astype(np.int32), near_far

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        tar_view, src_views, frame_id = self.metas[index]
        if self.split == 'train':
            if random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)
        scene_info = self.scene_info

        tar_img, tar_ext, tar_ixt, xywh, near_far = self.read_tar(tar_view, frame_id)
        src_inps, src_exts, src_ixts = self.read_src(src_views, frame_id)

        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        ret.update({'near_far': np.array([near_far, self.bkgd_near_far[tar_view]]).astype(np.float32)})
        ret.update({'bbox': xywh.astype(np.float32)})
        ret.update({'meta': {'scene': self.scene, 'tar_view': tar_view, 'frame_id': frame_id}})

        for i in range(cfg.enerf.cas_config.num):
            rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, np.ones_like(tar_img[..., 0]), i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32)})
        return ret

    def read_src(self, src_views, frame_id):
        inps, exts, ixts = [], [], []
        for src_view in src_views:
            img, ext, ixt = self.read_data(src_view, frame_id)
            inps.append(img * 2. - 1.)
            exts.append(ext)
            ixts.append(ixt)
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)

    def __len__(self):
        return len(self.metas)

