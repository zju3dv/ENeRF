import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
from lib.config import cfg
import numpy as np
import torch.nn.functional as F
import torch
import imageio
import os

class Visualizer:
    def __init__(self,):
        self.write_video = cfg.write_video
        self.imgs = []
        self.depths = []
        self.imgs_coarse = []
        os.system('mkdir -p {}'.format(cfg.result_dir))
        os.system('mkdir -p {}'.format(cfg.result_dir + '/imgs'))

    def visualize(self, output, batch):
        B, N_rays = batch['rays'].shape[:2]
        for b in range(B):
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            if 'fine_s_rgb_map_1' in output and cfg.render_static:
                img = output['fine_s_rgb_map_1'][b].reshape(h, w, 3).detach().cpu().numpy()
                depth = output['fine_s_depth_map_1'][b].reshape(h, w).detach().cpu().numpy()
            else:
                depth = output['depth_1'][b].reshape(h, w).detach().cpu().numpy()
                img = output['rgb_1'][b].reshape(h, w, 3).detach().cpu().numpy()
            img_coarse = output['rgb_0'][b].reshape(h, w, 3).detach().cpu().numpy()
            self.imgs_coarse.append(img_coarse)
            idx = batch['meta']['seq_id'][b].item()
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_rgb.png'.format(idx)), img)
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_rgb_coarse.png'.format(idx)), img_coarse)
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_dpt.png'.format(idx)), ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8))
            self.imgs.append(img)
            self.depths.append(depth)

    def summarize(self):
        imageio.mimwrite(os.path.join(cfg.result_dir, 'color_coarse.mp4'), self.imgs_coarse, fps=cfg.fps)
        imageio.mimwrite(os.path.join(cfg.result_dir, 'color.mp4'), self.imgs, fps=cfg.fps)
        d_min, d_max = np.array(self.depths).min(), np.array(self.depths).max()
        self.depths = [ (dpt - d_min)/(d_max-d_min) for dpt in self.depths ]
        imageio.mimwrite(os.path.join(cfg.result_dir, 'depth.mp4'), self.depths, fps=cfg.fps)



