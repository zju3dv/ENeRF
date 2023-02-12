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
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.loss_fn_vgg.cuda()
        os.system('mkdir -p ' + cfg.result_dir)

    def evaluate(self, output, batch):
        B, S, _, H, W = batch['src_inps'].shape
        for i in range(cfg.enerf.cas_config.num):
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            render_scale = cfg.enerf.cas_config.render_scale[i]
            h, w = int(H*render_scale), int(W*render_scale)
            pred_rgb = output[f'rgb_level{i}'].reshape(B, h, w, 3).detach().cpu().numpy()
            gt_rgb   = batch[f'rgb_{i}'].reshape(B, h, w, 3).detach().cpu().numpy()
            pred_depth = output[f'depth_level{i}'].reshape(B, h, w).detach().cpu().numpy()[..., None].repeat(3, -1)
            pred_depth -=  pred_depth.min()
            pred_depth /=  pred_depth.max()

            for b in range(B):
                if not batch['meta']['scene'][b]+f'_level{i}' in self.scene_psnrs:
                    self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'] = []
                # self.scene_lpips[batch['meta']['scene'][b]] = []
                if cfg.save_result and i == 1:
                    img = img_utils.horizon_concate(gt_rgb[b], pred_rgb[b])
                    img = img_utils.horizon_concate(img, pred_depth[b])
                    img_path = os.path.join(cfg.result_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                    imageio.imwrite(img_path, (img*255.).astype(np.uint8))

                psnr_item = psnr(gt_rgb[b], pred_rgb[b], data_range=1.)
                if i == cfg.enerf.cas_config.num-1:
                    self.psnrs.append(psnr_item)
                self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'].append(psnr_item)
                ssim_item = ssim(gt_rgb[b], pred_rgb[b], multichannel=True)
                if i == cfg.enerf.cas_config.num-1:
                    self.ssims.append(ssim_item)
                self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'].append(ssim_item)
                if cfg.eval_lpips:
                    gt, pred = torch.Tensor(gt_rgb[b])[None].permute(0, 3, 1, 2), torch.Tensor(pred_rgb[b])[None].permute(0, 3, 1, 2)
                    gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                    lpips_item = self.loss_fn_vgg(gt.cuda(), pred.cuda()).item()
                    if i == cfg.enerf.cas_config.num-1:
                        self.lpips.append(lpips_item)
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'].append(lpips_item)

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        ret.update({'ssim': np.mean(self.ssims)})
        if cfg.eval_lpips:
            ret.update({'lpips': np.mean(self.lpips)})
        print('='*30)
        for scene in self.scene_psnrs:
            if cfg.eval_lpips:
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} lpips:{:.3f}'.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene]), np.mean(self.scene_lpips[scene])))
            else:
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} '.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene])))
        print('='*30)
        print(ret)
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        if cfg.save_result:
            print('Save visualization results to: {}'.format(cfg.result_dir))
        return ret
