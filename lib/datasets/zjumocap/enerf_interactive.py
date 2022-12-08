import numpy as np
import os
from lib.config import cfg
import imp
import cv2
import random
import tqdm
from termcolor import colored
import imageio
import torch
if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)

from scipy import interpolate

from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.collate import default_convert, default_collate


from lib.utils.data_utils import to_cuda, add_batch
from lib.utils.net_utils import gen_rays_bbox, perf_timer
from lib.config import cfg

from typing import Mapping, Collection

timer = perf_timer(use_ms=True, logf=lambda x: print(colored(x, "green")), disabled=True)


class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

        self.metas = []
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        b,e,s = kwargs['frames']
        self.render_frames = np.arange(b, e)[::s].tolist()

        scene = kwargs['scene']
        self.scene = scene
        scene_info = {'ixts': [], 'exts': [], 'Ds': [], 'bbox': {}}
        scene_root = os.path.join(self.data_root, scene)
        annots = np.load(os.path.join(scene_root, 'annots.npy'), allow_pickle=True).item()
        self.annots = annots

        self.input_ratio = kwargs['input_ratio']

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
        b, e, s = kwargs['frames']
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
        self.scene_info = scene_info
        self.ixts = np.array(scene_info['ixts']).copy()
        self.exts = np.array(scene_info['exts'])
        self.cam_points = np.linalg.inv(self.exts)[:, :3, 3].astype(np.float32)
        self.cam_dirs = np.linalg.inv(self.exts)[:, :3, :3].astype(np.float32)

        self.ixts[:, :2] *= self.input_ratio
        self.ixt = np.mean(self.ixts, axis=0).astype(np.float32)

        self.input_h_w = (np.array([1024, 1024]) * self.input_ratio).astype(np.uint32).tolist()#
        H, W = self.input_h_w
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        self.X = torch.tensor(X).cuda()
        self.Y = torch.tensor(Y).cuda()
        self.XYZ = (XYZ @ np.linalg.inv(self.ixt).T).astype(np.float32)

        self.known_cams = np.arange(21)

        self.XYZ = torch.tensor(self.XYZ).cuda()
        self.ixt = torch.tensor(self.ixt).cuda()
        self.ixts = torch.tensor(self.ixts).cuda()
        self.exts = torch.tensor(self.exts).cuda()

        self.cache = {}
        print(colored('Preparing data...', 'yellow'))
        for i in tqdm.tqdm(self.render_frames):
            self.cache_data(i)  # preloading the data into self.cache

        self.split = kwargs['split']
        self.scene = kwargs['scene']
        self.kwargs = kwargs

    def read_data(self, scene, view, frame_id):
        scene_root = os.path.join(self.data_root, scene)
        scene_info = self.scene_info

        img_path = os.path.join(scene_root, self.annots['ims'][frame_id]['ims'][view])
        img = imageio.imread(img_path).astype(np.float32) / 255.

        mask_path = os.path.join(scene_root, 'mask', self.annots['ims'][frame_id]['ims'][view][:-4]+'.png') # TODO: mask_cihp
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

    def cache_data(self, frame):
        if frame in self.cache:
            return self.cache[frame]

        data_dict = {}
        data_dict['inps'] = []
        for cam in self.known_cams:
            data_dict['inps'].append((self.read_data(self.scene, cam, frame)[0] * 2. - 1.))
        data_dict['inps'] = np.stack(data_dict['inps'])
        data_dict['vertices'] = self.scene_info['bbox'][frame].astype(np.float32)
        vertices = data_dict['vertices']
        data_dict['bounds'] = np.concatenate([vertices.min(axis=0)[None], vertices.max(axis=0)[None]]).astype(np.float32)

        ret = default_convert(data_dict)
        ret = pin_memory(ret)
        self.cache[frame] = ret

    def build_rays(self, c2w, tar_ixt, scale):
        if scale != 1.:
            tar_ixt = tar_ixt.copy()
            tar_ixt[:2] *= scale
        H, W = self.input_h_w
        XYZ = self.XYZ @ c2w[:3, :3].T

        rays_o = c2w[:3, 3][None, None]
        rays_o = rays_o.repeat_interleave(H, dim=0)
        rays_o = rays_o.repeat_interleave(W, dim=1)
        rays = torch.cat((rays_o, XYZ), dim=-1)
        rays = torch.cat((rays, self.X[..., None], self.Y[..., None]), dim=-1)
        return rays.float().reshape(-1, 8), H, W

    def convert_data(self, data_dict, c2w, w2c):
        # this should be called after loading data from cache (or disk in case of miss)
        # deals with extrinsics related pre-computation, mainly ray generation and view selection
        timer.logtime("")

        ext = torch.tensor(w2c).cuda()
        c2w = torch.tensor(c2w).cuda()

        i = 1  # level 1
        scale = cfg.enerf.cas_config.render_scale[i]
        rays, H, W = self.build_rays(c2w, self.ixt, scale)

        timer.logtime("BUILD RAYS: {:.3f}")

        # TODO: use vertices to gen new mask in more detail
        bounds = data_dict['bounds'].cuda()
        rays_at_bbox = gen_rays_bbox(rays, bounds)
        # rays_at_bbox = gen_vertices_mask(H, W,
        #                                  self.ixt.detach().cpu().numpy(),
        #                                  ext.detach().cpu().numpy(),
        #                                  data_dict["vertices"].numpy(),
        #                                  self.faces.detach().cpu().numpy())  # Note: this will return cuda rays_at_bbox
        ret = {'tar_ext': ext.float(), 'tar_ixt': self.ixt, 'meta': {}}
        ret.update({f'rays_{i}': rays, f'mask_at_box': rays_at_bbox.reshape(self.input_h_w[0], self.input_h_w[1]).int()})

        ret['meta'].update({f'h_{i}': H, f'w_{i}': W})

        timer.logtime("GET MASK: {:.3f}")

        vertices = data_dict['vertices'].cuda() @ ext[:3, :3].T + ext[:3, 3:].T
        depth_min = vertices[:, 2].min()
        depth_max = vertices[:, 2].max()
        near_far = torch.tensor([max(depth_min.item(), 0.05), depth_max]).cuda()
        # near_far = torch.stack([near_far, torch.tensor([1., 8.]).cuda()])
        timer.logtime("GEN NEAR_FAR: {:.3f}")

        distances = np.linalg.norm(self.cam_points - c2w[:3, 3][None].cpu().numpy(), axis=-1)
        # distances = np.linalg.norm(self.cam_dirs - c2w[:3, :3][None].cpu().numpy(), axis=(-1, -2))
        argsorts = np.argsort(distances)
        near_views = argsorts[:cfg.enerf.test_input_views]

        timer.logtime("SELECT VIEWS: {:.3f}")

        # ret.update({'bbox': torch.tensor(np.array(xywhs).astype(np.int32)).cuda()})
        ret.update({
            'src_inps': data_dict['inps'][near_views].permute(0, 3, 1, 2).cuda(),
            'src_exts': self.exts[near_views],
            'src_ixts': self.ixts[near_views],
        })
        ret.update({'near_views': near_views})
        ret.update({'near_far': near_far})

        timer.logtime("MEM->GPU: {:.3f}")

        ret = add_batch(ret)  # add a batch dimension

        timer.logtime("ADD BATCH: {:.3f}")

        return ret


    def __getitem__(self, query):
        index, c2w, w2c = query
        data_dict = self.cache_data(index)
        ret = self.convert_data(data_dict, c2w, w2c)

        return ret  # return just loaded data



    def get_camera_up_front_center(self, index=0):
        """Return the worldup, front vectors and center of the camera
        Typically used to load camera parameters
        Extrinsic Matrix: leftmost column to rightmost column: world_down cross front, world_down, front
        [
            worldup, front, center (all column vectors)
        ]

        Args:
            index(int): which camera to load
        Returns:
            worldup(np.ndarray), front(np.ndarray), center(np.ndarray)
        """
        # TODO: loading from torch might be slow?
        ext = self.exts[index].detach().cpu().numpy()
        worldup, front, center = -ext.T[:3, 1], ext.T[:3, 2], -ext[:3, :3].T @ ext[:3, 3]
        return worldup, front, center

    def get_closest_camera(self, center):
        return np.argmin(np.linalg.norm(self.cam_points - center, axis=-1))

    def get_camera_tck(self, smoothing_term=0.0):
        """Return B-spline interpolation parameters for the camera
        TODO: Actually this should be implemented as a general interpolation function
        Reference get_camera_up_front_center for the definition of worldup, front, center
        Args:
            smoothing_term(float): degree of smoothing to apply on the camera path interpolation
        """
        # - R^t @ T = cam2world translation
        # TODO: loading from torch might be slow?
        exts = self.exts.detach().cpu().numpy()  # 21, 4, 4
        # TODO: load from cam_points to avoid repeated computation
        all_cens = -np.einsum("bij,bj->bi", exts[:, :3, :3].transpose(0, 2, 1), exts[:, :3, 3]).T
        all_fros = exts[:, 2, :3].T  # (3, 21)
        all_wups = -exts[:, 1, :3].T  # (3, 21)
        cen_tck, cen_u = interpolate.splprep(all_cens, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        fro_tck, fro_u = interpolate.splprep(all_fros, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        wup_tck, wup_u = interpolate.splprep(all_wups, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        return cen_tck, cen_u, fro_tck, fro_u, wup_tck, wup_u

    @property
    def n_cams(self):
        return len(self.known_cams)

    def __len__(self):
        return len(self.render_frames)
