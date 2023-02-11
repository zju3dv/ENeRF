import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .feature_net import FeatureNet, CNNRender
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from lib.config import cfg
from .nerf import NeRF

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.enerf.cas_config.num):
            for layer_idx in range(cfg.num_fg_layers):
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))))
                setattr(self, f'cost_reg_{i}_layer{layer_idx}', cost_reg_l)
                nerf_l = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
                setattr(self, f'nerf_{i}_layer{layer_idx}', nerf_l)
            cost_reg_l_bg = MinCostRegNet(int(32 * (2**(-i))))
            setattr(self, f'cost_reg_{i}_bg', cost_reg_l_bg)
            nerf_l_bg = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
            setattr(self, f'nerf_{i}_bg', nerf_l_bg)
        self.num_fg_layers = cfg.num_fg_layers
        # self.num_fg_layers = 1

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, nerf_model = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['nerf_model']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.enerf.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[level])
        up_feat_scale = cfg.enerf.cas_config.render_scale[level] / cfg.enerf.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.enerf.cas_config.render_scale[level]), int(W_O * cfg.enerf.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level) # B * N * S * (8+3+4)
        net_output = nerf_model(vox_feat, img_feat_rgb_dir)
        net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
        if cfg.enerf.cas_config.depth_inv[level]:
            return {'net_output': net_output, 'z_vals': 1./z_vals}
        else:
            return {'net_output': net_output, 'z_vals': z_vals}

    def batchify_rays(self, rays, **kwargs):
        all_ret = {}
        chunk = cfg.enerf.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:, i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret


    def forward_feat(self, x, feature_net):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward(self, batch):
        feats = self.forward_feat(batch['src_inps'], self.feature_net)
        ret = {}
        layers_inter = {}
        near_far_batch = batch['near_far'].clone()
        depth_, std_, near_far_ = None, None, None
        for i in range(cfg.enerf.cas_config.num):
            batch['near_far'] = near_far_batch
            ret_layers = []
            for layer_idx in range(self.num_fg_layers):
                x,y,w,h=(batch['bbox'][0][layer_idx] * cfg.enerf.cas_config.volume_scale[i]).int()
                x,y,w,h=x.item(),y.item(),w.item(),h.item()
                feature_volume, depth_values, near_far = utils.build_feature_volume_composite(
                    feats[f'level_{i}'],
                    batch,
                    D=cfg.enerf.cas_config.volume_planes[i],
                    layers_inter=layers_inter,
                    level=i,
                    layer_idx=layer_idx,
                    xywh=[x,y,w,h])
                B, D, H, W = depth_values.shape
                feature_volume, depth_prob = getattr(self, f'cost_reg_{i}_layer{layer_idx}')(feature_volume)
                feature_volume = F.pad(feature_volume, (x, W-x-w, y, H-y-h), 'constant')
                depth_prob = F.pad(depth_prob, (x, W-x-w, y, H-y-h), 'constant')
                depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
                layers_inter[f'depth_{i}_{layer_idx}'] = depth
                layers_inter[f'std_{i}_{layer_idx}'] = std
                layers_inter[f'near_far_{i}_{layer_idx}'] = near_far
                if cfg.enerf.cas_config.render_if[i]:
                    rays, xywh, BHW = utils.build_rays_composite(depth, std, batch, self.training, near_far, i, xywh=batch['bbox'][0][layer_idx])
                    im_feat_level = cfg.enerf.cas_config.render_im_feat_level[i]
                    ret_layer = self.batchify_rays(
                        rays=rays,
                        feature_volume=feature_volume,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'],
                        nerf_model=getattr(self, f'nerf_{i}_layer{layer_idx}'),
                        level=i)
                    ret_layers.append(ret_layer)
            batch['near_far'] = near_far_batch[:, -1]
            feature_volume_, depth_values_, near_far_ = utils.build_feature_volume(
                    feats[f'level_{i}'],
                    batch,
                    D=[16, 4][i],
                    depth=depth_,
                    std=std_,
                    near_far=near_far_,
                    level=i)
            feature_volume_, depth_prob_ = getattr(self, f'cost_reg_{i}_bg')(feature_volume_)
            depth_, std_ = utils.depth_regression(depth_prob_, depth_values_, i, batch)
            if cfg.enerf.cas_config.render_if[i]:
                rays_ = utils.build_rays(depth_, std_, batch, self.training, near_far_, i)
                im_feat_level = cfg.enerf.cas_config.render_im_feat_level[i]
                ret_layer = self.batchify_rays(
                    rays=rays_,
                    feature_volume=feature_volume_,
                    batch=batch,
                    im_feat=feats[f'level_{im_feat_level}'],
                    nerf_model=getattr(self, f'nerf_{i}_bg'),
                    level=i)
                ret_layers.append(ret_layer)
                h, w = batch['src_inps'].shape[-2:]
                ret_i = utils.raw2outputs_composite(ret_layers, batch, hw=[h,w], level=i, num_fg_layers=self.num_fg_layers)
                ret.update({key+f'_level{i}': ret_i[key] for key in ret_i})
                if torch.isnan(ret[f'rgb_level{i}']).any():
                    __import__('ipdb').set_trace()
        return ret
