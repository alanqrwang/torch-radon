# import odl
#
# geometry = odl.tomo.ConeFlatGeometry()
# operator = odl.tomo.RayTransform(space, geometry)
import torch
import torch_radon_cuda
from torch_radon_cuda import RaysCfg
import astra
import numpy as np
import time
import matplotlib.pyplot as plt


def benchmark_function(f, x, samples, warmup, sync=False):
    for _ in range(warmup):
        f(x)

    if sync:
        torch.cuda.synchronize()
    s = time.time()
    for _ in range(samples):
        f(x)
    if sync:
        torch.cuda.synchronize()
    e = time.time()

    return samples / (e - s)


geom_size = 128
det_count = geom_size
det_spacing = 2.0
s_dist = 128.0
d_dist = 128.0

vol_geom = astra.create_vol_geom(geom_size, geom_size, geom_size)

angles = np.linspace(-np.pi, np.pi, 128, False)
# proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 128, angles)
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, det_count, det_count, angles, s_dist, d_dist)

# Create a simple hollow cube phantom
cube = np.zeros((geom_size, geom_size, geom_size), dtype=np.float32)
cube[17:113, 17:113, 17:113] = 1
cube[33:97, 33:97, 33:97] = 0

print("Astra fps:", benchmark_function(lambda y: astra.create_sino3d_gpu(y, proj_geom, vol_geom), cube, 50, 5))
proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
print(proj_data.shape)
proj_data = proj_data.transpose(1, 0, 2)

# TORCH RADON
device = torch.device("cuda")

rays_cfg = RaysCfg(
    # width, height, depth
    geom_size, geom_size, geom_size,
    # det_count, det_spacing, det_count_z, det_spacing_z
    det_count, det_spacing, det_count, det_spacing,
    # n_angles, clip_to_circle
    len(angles), False,
    # source and detector distances
    s_dist, d_dist,
    # projection type
    2,
    # pitch, initial_z
    0.0, geom_size / 2
)

tex_cache = torch_radon_cuda.TextureCache(8)

x = torch.FloatTensor(cube).to(device).unsqueeze(0)
th_angles = torch.FloatTensor(angles).to(device)

print("TorchRadon fps:", benchmark_function(lambda y: torch_radon_cuda.forward(y, th_angles, tex_cache, rays_cfg), x, 50, 5, sync=True))

sinogram = torch_radon_cuda.forward(x, th_angles, tex_cache, rays_cfg)
torch.cuda.synchronize()
print(sinogram.size())

print(np.linalg.norm(proj_data - sinogram[0].cpu().numpy()) / np.linalg.norm(proj_data))

# print(sinogram[0, 64, 0, 64])

fig, ax = plt.subplots(1, 2)
ax = ax.ravel()
ax[0].imshow(proj_data[69])
ax[1].imshow(sinogram[0, 69].cpu().numpy())
plt.show()
