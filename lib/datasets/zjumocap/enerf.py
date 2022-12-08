import numpy as np
import os
from lib.config import cfg
import imp
import cv2
from lib.datasets import enerf_utils
import random
import imageio
from lib.utils import data_utils
# from .utils import sample_patch
if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_ratio = kwargs['input_ratio']
        self.scenes = [kwargs['scene']]
        self.build_metas(kwargs)


    def build_metas(self, config):
        self.scene_infos = {}
        self.metas = []
        self.annots = {}

        for scene in self.scenes:
            scene_info = {'ixts': [], 'exts': [], 'Ds': [], 'bbox': {}}
            scene_root = os.path.join(self.data_root, scene)
            annots = np.load(os.path.join(scene_root, 'annots.npy'), allow_pickle=True).item()

            self.scene_infos[scene] = scene_info
            self.annots[scene] = annots
            cam_len = annots['cams']['K'].__len__()

            for cam_id in range(cam_len):
                R = np.array(annots['cams']['R'][cam_id])
                T = np.array(annots['cams']['T'][cam_id]) / 1000.
                ext = np.concatenate((R, T), axis=1)
                bottom = np.zeros((1, 4))
                bottom[0, 3] = 1.
                ext = np.concatenate((ext, bottom))
                K = np.array(annots['cams']['K'][cam_id])
                D = np.array(annots['cams']['D'][cam_id])
                scene_info['exts'].append(ext.astype(np.float32))
                scene_info['ixts'].append(K.astype(np.float32))
                scene_info['Ds'].append(D.astype(np.float32))

            frame_len = len(annots['ims'])
            b, e, s = config['frames']
            e = e if e != -1 else frame_len

            for frame_id in np.arange(frame_len)[b:e:s]:
                vertices = np.load(f'{scene_root}/new_vertices/{frame_id+1}.npy') # Use SMPL vertices to compute bbox
                mi, ma = vertices.min(axis=0) - 0.1, vertices.max(axis=0) + 0.1
                vertices = np.array([[mi[0], mi[1], mi[2]],
                               [mi[0], mi[1], ma[2]],
                               [mi[0], ma[1], mi[2]],
                               [mi[0], ma[1], ma[2]],
                               [ma[0], mi[1], mi[2]],
                               [ma[0], mi[1], ma[2]],
                               [ma[0], ma[1], mi[2]],
                               [ma[0], ma[1], ma[2]]])
                scene_info['bbox'][frame_id] = vertices

            b, e, s = config['input_views']
            e = e if e != -1 else cam_len
            train_ids = np.arange(cam_len)[b:e:s]
            b, e, s = config['render_views']
            e = e if e != -1 else cam_len
            render_ids = np.arange(cam_len)[b:e:s]

            c2ws = np.linalg.inv(np.array(scene_info['exts']))

            train_cam_pos = c2ws[train_ids.tolist()][:, :3, 3]
            train_cam_dir = c2ws[train_ids.tolist()][:, :3, :3]

            b, e, s = config['frames']
            e = e if e != -1 else frame_len

            for render_id in render_ids:
                cam_pos = c2ws[:, :3, 3][render_id]
                cam_dir = c2ws[:, :3, :3][render_id]
                distance_dir = np.linalg.norm(train_cam_dir - cam_dir[None], axis=(1, 2))
                distance = np.linalg.norm(train_cam_pos - cam_pos[None], axis=-1)
                argsorts = distance.argsort()
                argsorts_dir = distance_dir.argsort()

                input_views_num = cfg.enerf.train_input_views[-1] + 1 if self.split == 'train' else cfg.enerf.test_input_views
                nearest_ids = argsorts_dir[:2*input_views_num]

                if render_id not in train_ids or self.split == 'test':
                    src_views = [train_ids[i] for i in argsorts[:input_views_num] if i in nearest_ids]
                else:
                    src_views = [train_ids[i] for i in argsorts[1:input_views_num+1] if i in nearest_ids]
                if len(src_views) < input_views_num:
                    __import__('ipdb').set_trace()
                self.metas += [(scene, render_id, src_views, frame_id) for frame_id in (np.arange(frame_len))[b:e:s]]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views, frame_id = self.metas[index]
        if self.split == 'train':
            if random.random() < 0.05:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)
        tar_img, tar_ext, tar_ixt, tar_msk, near_far, mask_at_box = self.read_tar(scene, tar_view, frame_id)
        src_inps, src_exts, src_ixts = self.read_src(scene, src_views, frame_id)

        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'mask_at_box': mask_at_box})
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': frame_id}})

        for i in range(cfg.enerf.cas_config.num):
            rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_msk, i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb, f'msk_{i}': msk})
        return ret

    def read_data(self, scene, view, frame_id):
        scene_root = os.path.join(self.data_root, scene)
        scene_info = self.scene_infos[scene]

        img_path = os.path.join(scene_root, self.annots[scene]['ims'][frame_id]['ims'][view])
        img = imageio.imread(img_path).astype(np.float32) / 255.

        mask_path = os.path.join(scene_root, 'mask', self.annots[scene]['ims'][frame_id]['ims'][view][:-4]+'.png') # TODO: mask_cihp
        mask = imageio.imread(mask_path)
        mask = (mask != 0).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        mask = cv2.dilate(mask.copy(), kernel)

        ext, ixt, D = scene_info['exts'][view], scene_info['ixts'][view].copy(), scene_info['Ds'][view]
        img = cv2.undistort(img, ixt, D)
        mask = cv2.undistort(mask, ixt, D)

        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_NEAREST)
            ixt[:2] *= self.input_ratio

        img[mask == 0] = 0.
        return img, mask, ext, ixt



    def read_tar(self, scene, tar_view, frame_id):
        scene_info = self.scene_infos[scene]
        img, mask, ext, ixt = self.read_data(scene, tar_view, frame_id)
        bbox_scene = scene_info['bbox'][frame_id]
        bbox_scene = np.concatenate([bbox_scene, np.ones_like(bbox_scene[..., :1])], axis=-1)
        bbox_scene = bbox_scene @ ext.T
        depth_ranges = [max(bbox_scene[..., 2].min(), 0.1), bbox_scene[..., 2].max()]
        bound_mask = data_utils.get_bound_2d_mask(bbox_scene, ixt, img.shape[0], img.shape[1])
        return img, ext, ixt, mask, depth_ranges, bound_mask

    def read_src(self, scene, src_views, frame_id):
        inps, exts, ixts = [], [], []
        for src_view in src_views:
            img, _, ext, ixt = self.read_data(scene, src_view, frame_id)
            inps.append(2 * img - 1.)
            exts.append(ext)
            ixts.append(ixt)
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)

    def build_rays(self, dataset, tar_img, tar_ext, tar_ixt, tar_msk, level):
        scale = cfg.cas_config.render_scale[level]
        if scale != 1.:
            tar_img = cv2.resize(tar_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            tar_msk = cv2.resize(tar_msk, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            tar_ixt = tar_ixt.copy()
            tar_ixt[:2] *= scale

        H, W = tar_img.shape[:2]
        c2w = np.linalg.inv(tar_ext)
        if self.split == 'train' and not cfg.cas_config.train_img[level]:
            if cfg.sample_on_edge:
                msk_sample = np.zeros_like(tar_msk)
                border = 11
                kernel = np.ones((border, border), np.uint8)
                mask_erode = cv2.erode(tar_msk.copy(), kernel)
                mask_dilate = cv2.dilate(tar_msk.copy(), kernel)
                msk_sample[(mask_dilate - mask_erode) == 1] = 1
                num_fg_rays = int(min(cfg.cas_config.num_rays[level]*0.5, msk_sample.sum()))
                non_zero = msk_sample.nonzero()
                permutation = np.random.permutation(msk_sample.sum())[:num_fg_rays].astype(np.int32)
                X_, Y_ = non_zero[1][permutation], non_zero[0][permutation]
            elif cfg.sample_on_mask:
                msk_sample = tar_msk
                num_fg_rays = int(min(cfg.cas_config.num_rays[level]*0.75, tar_msk.sum()*0.95))
                non_zero = msk_sample.nonzero()
                permutation = np.random.permutation(tar_msk.sum())[:num_fg_rays].astype(np.int32)
                X_, Y_ = non_zero[1][permutation], non_zero[0][permutation]
            else:
                num_fg_rays = 0
                msk_sample = np.zeros_like(tar_msk)
            num_rays = cfg.cas_config.num_rays[level] - num_fg_rays
            X = np.random.randint(low=0, high=W, size=num_rays)
            Y = np.random.randint(low=0, high=H, size=num_rays)
            if num_fg_rays > 0:
                X = np.concatenate([X, X_]).astype(np.int32)
                Y = np.concatenate([Y, Y_]).astype(np.int32)
            if cfg.cas_config.num_patchs[level] > 0:
                X_, Y_ = sample_patch(cfg.cas_config.num_patchs[level], cfg.cas_config.patch_size[level], H, W, msk_sample)
                X = np.concatenate([X, X_]).astype(np.int32)
                Y = np.concatenate([Y, Y_]).astype(np.int32)
            num_rays = len(X)
            rays_o = c2w[:3, 3][None].repeat(num_rays, 0)
            XYZ = np.concatenate((X[:, None], Y[:, None], np.ones_like(X[:, None])), axis=-1)
            XYZ = XYZ @ (np.linalg.inv(tar_ixt).T @ c2w[:3, :3].T)
            rays = np.concatenate((rays_o, XYZ, X[..., None], Y[..., None]), axis=-1)
            rgb = tar_img[Y, X]
            msk = tar_msk[Y, X]
        else:
            rays_o = c2w[:3, 3][None, None]
            X, Y = np.meshgrid(np.arange(W), np.arange(H))
            XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
            XYZ = XYZ @ (np.linalg.inv(tar_ixt).T @ c2w[:3, :3].T)
            rays_o = rays_o.repeat(H, axis=0)
            rays_o = rays_o.repeat(W, axis=1)
            rays = np.concatenate((rays_o, XYZ, X[..., None], Y[..., None]), axis=-1)
            rgb = tar_img
            msk = tar_msk
        return rays.astype(np.float32).reshape(-1, 8), rgb.reshape(-1, 3), msk.reshape(-1)

    def __len__(self):
        return len(self.metas)
