import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_radon import Radon, RadonFanbeam

device = torch.device('cuda')

img = np.load("examples/phantom.npy")
# img = np.ones((16, 16), dtype=np.float32)
image_size = img.shape[0]
n_angles = image_size

# Instantiate Radon transform. clip_to_circle should be True when using filtered backprojection.
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
radon = Radon(image_size, angles)

with torch.no_grad():
    x = torch.FloatTensor(img).to(device).unsqueeze(0).repeat(8, 1, 1).to(device) #.half()

    sinogram = radon.forward(x)
    # filtered_sinogram = radon.filter_sinogram(sinogram, "ram-lak")
    torch.cuda.synchronize()
    print("\n\n")
    print(sinogram.size())
    bp = radon.backprojection(sinogram)

plt.imshow(sinogram[5].cpu().float().numpy())
# # Show results
# titles = ["Original Image", "Sinogram", "Backprojection"]
# show_images([x, sinogram, bp], titles, keep_range=False)
plt.show()