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
        B, S, _, H, W = batch['src_inps'].shape
        i = cfg.enerf.cas_config.num - 1
        render_scale = cfg.enerf.cas_config.render_scale[i]
        h, w = int(H*render_scale), int(W*render_scale)
        assert(B == 1)
        pred_rgb = output[f'rgb_level{i}'].reshape(h, w, 3).detach().cpu().numpy()
        depth = output[f'depth_level{i}'].reshape(h, w).detach().cpu().numpy()
        crop_h, crop_w = int(h * 0.1), int(w * 0.1)
        pred_rgb = pred_rgb[crop_h:, crop_w:-crop_w]
        depth = depth[crop_h:, crop_w:-crop_w]
        self.imgs.append(pred_rgb)
        self.depths.append(depth)
        if cfg.save_result:
            frame_id = batch['meta']['frame_id'][0].item()
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_rgb.jpg'.format(frame_id)), pred_rgb)
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_dpt.jpg'.format(frame_id)), ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8))

    def summarize(self):
        imageio.mimwrite(os.path.join(cfg.result_dir, 'color.mp4'), self.imgs, fps=cfg.fps)
        d_min, d_max = np.array(self.depths).min(), np.array(self.depths).max()
        self.depths = [ (dpt - d_min)/(d_max-d_min) for dpt in self.depths ]
        imageio.mimwrite(os.path.join(cfg.result_dir, 'depth.mp4'), self.depths, fps=cfg.fps)



