import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
import numpy as np
import torch.nn.functional as F
from lib.config import cfg
import cv2
import torch
import imageio
from skimage.metrics import peak_signal_noise_ratio as psnr

class Visualizer:
    def __init__(self,):
        self.HW = [512, 512]

    def visualize(self, output, batch):
        B = 1
        for b in range(B):
            # H, W = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            # H, W = self.HW
            _, _, _, H, W = batch['src_inps'].shape
            i = cfg.enerf.cas_config.num - 1
            # gt_img = batch[f'rgb_{i}'][b].reshape(H, W, 3).detach().cpu().numpy()
            pred_img = output[f'rgb_level{i}'][b].reshape(H, W, 3)
            # imageio.imwrite('test.png', pred_img.detach().cpu().numpy())
            ret = {'pred': pred_img}
            if 'vis_ret' in output and False:
                seg = (output['vis_ret']['layer_0_weight'] > output['vis_ret']['layer_3_weight'])[b].float()[..., None].repeat(1, 1, 3)
                depth = output['vis_ret']['depth'][b][..., None].repeat(1, 1, 3)
                bbox = batch['masks'][b, 0].float()[..., None].repeat(1, 1, 3)
                ret.update({'seg': seg, 'depth': depth, 'bbox': bbox})
            # src_inps = (batch['src_inps'][b] * 0.5 + 0.5).detach().cpu()
            # idx = 0
            # for src_inp in src_inps:
                # cv2.imshow(f'src_{idx}', ((src_inp.permute(1, 2, 0).numpy()[..., [2,1,0]])*255).astype(np.uint8))
                # cv2.waitKey(1)
                # idx += 1
            # ret.update({'bbox': batch['masks'][0][0][..., None].repeat(1, 1, 3)})

            return ret
            # psnr_item = psnr(gt_img, pred_img, data_range=1.)

            # print(psnr_item)
            # plt.imshow(np.concatenate([gt_img, pred_img], axis=1))
            # plt.show()

