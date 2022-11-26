import numpy as np
import torch
from tqdm import tqdm
import trimesh
from skimage import measure

def extract_mesh(queryfn, level, bbox, output_path='test.ply', N=256):
    bbox = np.array(bbox).reshape((2, 3))

    voxel_grid_origin = np.mean(bbox, axis=0)
    volume_size = bbox[1] - bbox[0]
    s = volume_size[0]

    overall_index = np.arange(0, N ** 3, 1).astype(np.int)
    xyz = np.zeros([N ** 3, 3])

    # transform first 3 columns
    # to be the x, y, z index
    xyz[:, 2] = overall_index % N
    xyz[:, 1] = (overall_index / N) % N
    xyz[:, 0] = ((overall_index / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    xyz[:, 0] = (xyz[:, 0] * (s/(N-1))) + bbox[0][0]
    xyz[:, 1] = (xyz[:, 1] * (s/(N-1))) + bbox[0][1]
    xyz[:, 2] = (xyz[:, 2] * (s/(N-1))) + bbox[0][2]

    xyz = torch.from_numpy(xyz).float()

    batch_size = 8192
    density = []
    for i in tqdm(range(N ** 3 // batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        density.append(queryfn(xyz[start: end].cuda())[..., 0].detach().cpu())

    density = torch.cat(density, dim=-1)
    density = density.view(N, N, N)
    vertices, faces, normals, _ = measure.marching_cubes_lewiner(density.numpy(), level=level, spacing=[float(v) / N for v in volume_size])
    vertices += voxel_grid_origin
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)


