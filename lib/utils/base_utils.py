import pickle
import os
import numpy as np
import cv2
import time
from termcolor import colored
import importlib
import torch.distributed as dist
import math

class perf_timer:
    def __init__(self, msg="Elapsed time: {}s", logf=lambda x: print(colored(x, 'yellow')), sync_cuda=True, use_ms=False, disabled=False):
        self.logf = logf
        self.msg = msg
        self.sync_cuda = sync_cuda
        self.use_ms = use_ms
        self.disabled = disabled

        self.loggedtime = None

    def __enter__(self,):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.logtime(self.msg)

    def logtime(self, msg=None, logf=None):
        if self.disabled:
            return
        # SAME CLASS, DIFFERENT FUNCTIONALITY, is this good?
        # call the logger for timing code sections
        if self.sync_cuda:
            torch.cuda.synchronize()

        # always remember current time
        prev = self.loggedtime
        self.loggedtime = time.perf_counter()

        # print it if we've remembered previous time
        if prev is not None and msg:
            logf = logf or self.logf
            diff = self.loggedtime-prev
            diff *= 1000 if self.use_ms else 1
            logf(msg.format(diff))

        return self.loggedtime

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bbox_2d(bbox, K, RT):
    pts = np.array([[bbox[0, 0], bbox[0, 1], bbox[0, 2]],
                    [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
                    [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
                    [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
                    [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
                    [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
                    [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
                    [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
                    ])
    pts_2d = project(pts, K, RT)
    return [pts_2d[:, 0].min(), pts_2d[:, 1].min(), pts_2d[:, 0].max(), pts_2d[:, 1].max()]


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def load_object(module_name, module_args, **extra_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**extra_args, **module_args)
    return obj



def get_indices(length):
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    num_samples = int(math.ceil(length * 1.0 / num_replicas))
    total_size = num_samples * num_replicas
    indices = np.arange(length).tolist()
    indices += indices[: (total_size - len(indices))]
    offset = num_samples * rank
    indices = indices[offset:offset+num_samples]
    return indices


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        if dct is not None:
            for key, value in dct.items():
                if hasattr(value, 'keys'):
                    value = DotDict(value)
                # self[key] = value



