import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2


class Evaluator:

    def __init__(self,):
        self.psnrs = []
        self.psnrs_0 = []
        os.system('mkdir -p ' + cfg.result_dir)

    def evaluate(self, output, batch):
        B, N_rays = batch['rays'].shape[:2]
        for b in range(B):
            gt_rgb = batch['rgb'][b].reshape(-1, 3).detach().cpu().numpy()
            pred_rgb = output['rgb_1'][b].detach().cpu().numpy()
            self.psnrs.append(psnr(pred_rgb, gt_rgb, data_range=1.))
            pred_rgb = output['rgb_0'][b].detach().cpu().numpy()
            psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
            self.psnrs_0.append(psnr_item)
            if cfg.save_result:
                h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
                gt_rgb = batch['rgb'][b].reshape(h, w, 3).detach().cpu().numpy()
                pred_rgb_coarse = output['rgb_0'][b].reshape(h, w, 3).detach().cpu().numpy()
                pred_rgb_fine = output['rgb_1'][b].reshape(h, w, 3).detach().cpu().numpy()
                save_path = os.path.join(cfg.result_dir, 'view{:06d}'.format(batch['meta']['idx'][b].item()))
                save_path = save_path + '_{}.jpg'
                imageio.imwrite(save_path.format('gt'), gt_rgb)
                imageio.imwrite(save_path.format('coarse'), pred_rgb_coarse)
                imageio.imwrite(save_path.format('fine'), pred_rgb_fine)

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        if len(self.psnrs_0) != 0:
            ret.update({'psnr_0': np.mean(self.psnrs_0)})
            self.psnrs_0 = []
        print(ret)
        self.psnrs = []
        return ret
