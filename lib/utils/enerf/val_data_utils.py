import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from lib.networks.enerf.utils import unpreprocess

def validate(batch):
    B = len(batch['tar_img'])
    num_points = 10
    batch_src_inps = unpreprocess(batch['src_inp']).cpu().numpy()
    for b in range(B):
        rgb = batch['tar_img'][b].cpu().numpy() # rgb
        gray = cv2.cvtColor((rgb*255.).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detectAndCompute(gray, None)[0]
        points = [keypoint.pt for keypoint in keypoints]
        points = np.stack(random.sample(points, num_points))
        near, far = batch['near_far'][b][0].item(), batch['near_far'][b][1].item()
        points_near = np.concatenate((points, near * np.ones_like(points[:, :1])), axis=-1)
        points_far = np.concatenate((points, far * np.ones_like(points[:, :1])), axis=-1)

        src_inps = batch_src_inps[b]
        S = len(src_inps)

        ax = plt.subplot(1, 1+S, 1)
        ax.axis('off')
        ax.set_title('1')
        plt.imshow(rgb)
        for i in range(len(points)):
            plt.plot(points[i, 0][None], points[i, 1][None], '.')

        for s in range(S):
            points_near_s = transform(points_near.copy(), batch, b, s)
            points_far_s = transform(points_far.copy(), batch, b, s)
            lines = []
            for point_near, point_far in zip(points_near_s, points_far_s):
                lines.append(np.concatenate((point_near[:2][None], point_far[:2][None])))

            ax = plt.subplot(1, 1+S, s+2)
            ax.axis('off')
            ax.set_title('{}_{}'.format(1+S, s+1))
            src_inp = batch_src_inps[b][s].transpose(1, 2, 0)
            plt.imshow(src_inp)
            for i in range(len(lines)):
                plt.plot(lines[i][:, 0], lines[i][:, 1])

        plt.subplots_adjust(left=0.,bottom=0.,top=1.,right=1.,hspace=0.,wspace=0.)
        plt.show()

def transform(points, batch, b, s):
    tar_ext = batch['tar_ext'][b].cpu().numpy()
    tar_ixt = batch['tar_ixt'][b].cpu().numpy()
    src_ext = batch['src_ext'][b, s].cpu().numpy()
    src_ixt = batch['src_ixt'][b, s].cpu().numpy()
    c2w = np.linalg.inv(tar_ext)
    points[..., :2] = points[..., :2] * points[..., 2:]
    points = points @ np.linalg.inv(tar_ixt).transpose(-1, -2)
    points = np.concatenate((points, np.ones_like(points[..., :1])), axis=-1)
    points = points @ np.linalg.inv(tar_ext).transpose(-1, -2)
    points = points @ src_ext.transpose(-1, -2)
    points = points[..., :3] @ src_ixt.transpose(-1, -2)
    points[..., :2] = points[..., :2] / points[..., 2:]
    return points
