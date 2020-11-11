#include "texture.h"

template<typename T>
void radon_forward_cuda(
        const T *x, const float *angles, T *y,
        TextureCache &tex_cache, const RaysCfg &rays_cfg, const int batch_size, const int device
);

void radon_forward_cuda_3d(
        const float *x, const float *angles, float *y,
        TextureCache &tex_cache, const RaysCfg &rays_cfg, const int batch_size, const int device
);


