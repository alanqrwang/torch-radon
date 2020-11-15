import torch
import torch_radon_cuda
from torch_radon import RadonConeFlat
import astra
import numpy as np
import time
import matplotlib.pyplot as plt

import odl


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


dtype = np.float32

size = 128
n_angles = 128
geom_size = size
s_dist = size
d_dist = size
pitch = geom_size / 4

shape = [size, size, size]
space = odl.uniform_discr([-size / 2] * 3, [size / 2] * 3, shape, dtype=dtype)
dpart = odl.uniform_partition([-size] * 2, [size] * 2, [size] * 2)

apart = odl.uniform_partition(-np.pi, np.pi, n_angles)

mi = -np.pi + 2*np.pi / (2 * n_angles)
ma = np.pi - 2*np.pi / (2 * n_angles)
angles = np.linspace(mi, ma, n_angles, endpoint=True)
# angles = np.linspace(-np.pi, np.pi, 2*n_angles, True)
# angles = (angles[::2] + angles[1::2]) / 2

# print(apart.points())
# print(angles)

geometry = odl.tomo.ConeFlatGeometry(apart, dpart, s_dist, d_dist, pitch=pitch, offset_along_axis=0.0)
operator = odl.tomo.RayTransform(space, geometry)

cube = np.zeros((geom_size, geom_size, geom_size), dtype=np.float32)
cube[17:113, 17:113, 17:113] = 1
cube[33:97, 33:97, 33:97] = 0
# cube += 0.1*np.random.randn(*cube.shape)
cube = cube.astype(np.float32)

# print("ODL fps:", benchmark_function(lambda y: operator(y), cube, 5, 1, sync=True))
#
proj_data = np.asarray(operator(cube)).transpose(0, 2, 1)
# print(proj_data.shape)

# fig, ax = plt.subplots(1, 2)
# ax = ax.ravel()
# ax[0].imshow(proj_data[69])
# ax[1].imshow(sinogram[0, 69].cpu().numpy())
# plt.show()

#
#

#
#
# geom_size = 128
det_count = geom_size
det_spacing = 2.0
# s_dist = 128.0
# d_dist = 2*128.0
#
vol_geom = astra.create_vol_geom(cube.shape[1], cube.shape[2], cube.shape[0])

# proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 128, angles)
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, det_count, det_count, angles, s_dist, d_dist)

# Create a simple hollow cube phantom
# cube = np.zeros((geom_size, geom_size, geom_size), dtype=np.float32)
# cube[17:113, 17:113, 17:113] = 1
# cube[33:97, 33:97, 33:97] = 0

print("Astra fps:", benchmark_function(lambda y: astra.create_sino3d_gpu(y, proj_geom, vol_geom), cube, 5, 1))
# proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
# print("Astra sinogram shape", proj_data.shape)

# vol_geom = astra.create_vol_geom(geom_size, geom_size, geom_size)
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
# cfg = astra.astra_dict('BP3D_CUDA')
# cfg['ReconstructionDataId'] = rec_id
# cfg['ProjectionDataId'] = proj_id
#
# # Create the algorithm object from the configuration structure
# alg_id = astra.algorithm.create(cfg)
# astra.algorithm.run(alg_id, 1)
# print("Astra BP fps:", benchmark_function(lambda y: astra.algorithm.run(alg_id, 1), cube, 5, 1))
#
# astra_bp = astra.data3d.get(rec_id)
# print("Astra bp", astra_bp.shape)

# proj_data = proj_data.transpose(1, 0, 2)
#
# TORCH RADON
device = torch.device("cuda")

radon = RadonConeFlat(cube.shape, angles, s_dist, d_dist, det_count=geom_size, det_spacing=det_spacing, pitch=pitch)

x = torch.FloatTensor(cube).to(device).unsqueeze(0) #.repeat(4, 1, 1, 1).half()
print("X size", x.size())
th_angles = torch.FloatTensor(angles).to(device)

print("TorchRadon fps:", x.size(0)*benchmark_function(lambda z: radon.forward(z), x, 5, 1, sync=True))

sinogram = radon.forward(x).float()
print("TorchRadon fps:", x.size(0)*benchmark_function(lambda z: radon.backward(z), sinogram, 5, 1, sync=True))
y = radon.backward(sinogram).float()
print("TorchRadon sinogram size", sinogram.size())
print("TorchRadon BP size", y.size())
torch.cuda.synchronize()
# print(sinogram.size(), y.size())

print(np.linalg.norm(proj_data - sinogram[0].cpu().numpy()) / np.linalg.norm(proj_data))
# print(np.linalg.norm(astra_bp - y[0].cpu().numpy()) / np.linalg.norm(astra_bp))

# print(sinogram[0, 64, 0, 64])

fig, ax = plt.subplots(3, 3)
ax = ax.ravel()
ax[0].imshow(proj_data[0])
ax[1].imshow(sinogram[0, 0].cpu().numpy())
ax[2].imshow(np.abs(sinogram[0, 0].cpu().numpy() - proj_data[0]))
ax[3].imshow(proj_data[64])
ax[4].imshow(sinogram[0, 64].cpu().numpy())
ax[5].imshow(np.abs(sinogram[0, 64].cpu().numpy() - proj_data[64]))
ax[6].imshow(proj_data[-1])
ax[7].imshow(sinogram[0, -1].cpu().numpy())
ax[8].imshow(np.abs(sinogram[0, -1].cpu().numpy() - proj_data[-1]))
# ax[0].imshow(astra_bp[0])
# ax[1].imshow(y[0, 0].cpu().numpy())
# ax[2].imshow(np.abs(y[0, 0].cpu().numpy() - astra_bp[0]))
# ax[3].imshow(astra_bp[64])
# ax[4].imshow(y[0, 64].cpu().numpy())
# ax[5].imshow(np.abs(y[0, 64].cpu().numpy() - astra_bp[64]))
# ax[6].imshow(astra_bp[-1])
# ax[7].imshow(y[0, -1].cpu().numpy())
# ax[8].imshow(np.abs(y[0, -1].cpu().numpy() - astra_bp[0]))
# ax[2].imshow(astra_bp[0])
# ax[3].imshow(y[0, 0].cpu().numpy())
# ax[4].imshow(astra_bp[64])
# ax[5].imshow(y[0, 64].cpu().numpy())
plt.show()
