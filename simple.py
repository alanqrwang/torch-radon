from torch_radon.parameter_classes import Volume
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_radon import Radon, Projection, ExecCfg

dy = 16
dx = 32
device = torch.device("cuda")

img = np.zeros((256, 256), dtype=np.float32)
img[128-16+dy:128+16+dy, 128-16+dx:128+16+dx] = 1.0
vol = Volume.create_2d(*img.shape)

img_d = np.zeros((256, 256), dtype=np.float32)
img_d[128-16:128+16, 128-16:128+16] = 1.0
vol_d = Volume.create_2d(*img_d.shape, dy, dx)

cropped_h = img.shape[0] - abs(dy)
cropped_w = img.shape[1] - abs(dx)
cropped_img = img[max(dy, 0):max(dy, 0) + cropped_h, max(dx, 0):max(dx, 0) + cropped_w] 
cropped_img_d = img_d[max(-dy, 0):max(-dy, 0) + cropped_h, max(-dx, 0):max(-dx, 0) + cropped_w]
print(np.linalg.norm(cropped_img-cropped_img_d))

angles = np.linspace(0, np.pi, 128, endpoint=False)
projection = Projection.parallel_beam(256)

radon = Radon(angles, vol, projection)
radon_d = Radon(angles, vol_d, projection)

with torch.no_grad():
    x = torch.FloatTensor(img).to(device)
    x_d = torch.FloatTensor(img_d).to(device)

    sinogram = radon.forward(x)
    sinogram_d = radon_d.forward(x_d)

    bp = radon.backprojection(sinogram)
    bp_d = radon_d.backprojection(sinogram_d)

cropped_bp = bp[max(dy, 0):max(dy, 0) + cropped_h, max(dx, 0):max(dx, 0) + cropped_w] 
cropped_bp_d = bp_d[max(-dy, 0):max(-dy, 0) + cropped_h, max(-dx, 0):max(-dx, 0) + cropped_w]
print(torch.norm(cropped_bp - cropped_bp_d) / torch.norm(cropped_bp))

plt.imshow(cropped_bp.cpu().float().numpy())
plt.figure()
plt.imshow(cropped_bp_d.cpu().float().numpy())

plt.show()

# device = torch.device('cuda')

# img = np.load("examples/phantom.npy")
# img = img[::2,]
# padded = np.pad(img, ((128, 128), (0,0)))

# image_size = max(img.shape)
# n_angles = image_size

# angles = np.linspace(0, np.pi, n_angles, endpoint=False)
# #projection = Projection.parallel_beam(image_size)
# projection = Projection.fanbeam(image_size, image_size, image_size)

# radon = Radon(angles, img.shape, projection)
# radon_p = Radon(angles, padded.shape, projection)

# with torch.no_grad():
#     x = torch.FloatTensor(img).to(device).to(device) #.half()
#     xp = torch.FloatTensor(padded).to(device).to(device) #.half()

#     sinogram = radon.forward(x)
#     sinogram_p = radon_p.forward(xp)

#     bp = radon.backprojection(sinogram)
#     bp_p = radon_p.backprojection(sinogram_p)

# print(torch.norm(sinogram_p - sinogram) / torch.norm(sinogram))

# print(bp.size(), bp_p.size())
# bp_p_cropped = bp_p[128:-128]
# print(torch.norm(bp_p_cropped - bp) / torch.norm(bp))
# # plt.imshow(sinogram.cpu().float().numpy())
# # plt.figure()
# # plt.imshow(sinogram_p.cpu().float().numpy())

# plt.imshow(bp.cpu().float().numpy())
# plt.figure()
# plt.imshow(bp_p.cpu().float().numpy())

# # # # Show results
# # titles = ["Original Image", "Sinogram", "Backprojection"]
# # show_images([x, sinogram, bp], titles, keep_range=False)
# plt.show()