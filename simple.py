import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_radon import Radon, Projection, ExecCfg

device = torch.device('cuda')

img = np.load("examples/phantom.npy")
img = img[::2, ::2]
# img = np.ones((16, 16), dtype=np.float32)
image_size = img.shape[0]
print(img.shape)
n_angles = image_size

# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
projection = Projection.fanbeam(image_size, image_size, image_size)
radon = Radon(angles, image_size, projection)

with torch.no_grad():
    x = torch.FloatTensor(img).to(device).unsqueeze(0).repeat(32, 1, 1).to(device) #.half()

    sinogram = radon.forward(x)
    # filtered_sinogram = radon.filter_sinogram(sinogram, "ram-lak")
    bp = radon.backprojection(sinogram)
    # bp = radon.backprojection(sinogram, exec_cfg=ExecCfg(32, 8, 1, 4))
    # bp = radon.backprojection(sinogram, exec_cfg=ExecCfg(32, 16, 1, 4))
    # bp = radon.backprojection(sinogram, exec_cfg=ExecCfg(32, 32, 1, 4))

plt.imshow(bp[5].cpu().float().numpy())
# # # Show results
# titles = ["Original Image", "Sinogram", "Backprojection"]
# show_images([x, sinogram, bp], titles, keep_range=False)
plt.show()