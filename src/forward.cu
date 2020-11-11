#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#include "utils.h"
#include "texture.h"

template<bool parallel_beam, int channels, bool clip_to_circle, typename T>
__global__ void
radon_forward_kernel(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                     RaysCfg cfg) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * channels;

    if (angle_id < cfg.n_angles && ray_id < cfg.det_count) {
        
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }


        // compute ray
        float v, sx, sy, ex, ey;
        if (parallel_beam) {
            v = cfg.height / 2.0;
            sx = (ray_id - cfg.det_count / 2.0f + 0.5f) * cfg.det_spacing;
            sy = 0.71f * cfg.height;
            ex = sx;
            ey = -sy;
        } else {
            v = cfg.height / 2.0;
            sy = cfg.s_dist;
            sx = 0.0f;
            ey = -cfg.d_dist;
            ex = (ray_id - cfg.det_count / 2.0f + 0.5f) * cfg.det_spacing;
        }

        
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;


        if (cfg.clip_to_circle) {
            // clip rays to circle (to reduce memory reads)
            const float radius = cfg.det_count / 2.0f;
            const float a = rdx * rdx + rdy * rdy;
            const float b = rsx * rdx + rsy * rdy;
            const float c = rsx * rsx + rsy * rsy - radius * radius;

            // min_clip to 1 to avoid getting empty rays
            const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
            const float alpha_s = (-b - delta_sqrt) / a;
            const float alpha_e = (-b + delta_sqrt) / a;

            rsx += rdx * alpha_s + v;
            rsy += rdy * alpha_s + v;
            rdx *= (alpha_e - alpha_s);
            rdy *= (alpha_e - alpha_s);
        } else {
            // clip to square (to reduce memory reads)
            const float alpha_x_m = (-v - rsx)/rdx;
            const float alpha_x_p = (v - rsx)/rdx;
            const float alpha_y_m = (-v -rsy)/rdy;
            const float alpha_y_p = (v - rsy)/rdy;
            const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
            const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

            if(alpha_s > alpha_e){
                #pragma unroll
                for (int b = 0; b < channels; b++) {
                    output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] = 0.0f;
                }
                return;
            }

            rsx += rdx*alpha_s + v;
            rsy += rdy*alpha_s + v;
            rdx *= (alpha_e - alpha_s);
            rdy *= (alpha_e - alpha_s);
        }
        
        const uint n_steps = __float2uint_ru(hypot(rdx, rdy));
        const float vx = rdx / n_steps;
        const float vy = rdy / n_steps;
        const float n = hypot(vx, vy);

        for (uint j = 0; j <= n_steps; j++) { //changing j and n_steps to int makes everything way slower (WHY???)
            if (channels == 1) {
                accumulator[0] += tex2DLayered<float>(texture, rsx, rsy, blockIdx.z);
            } else {
                float4 read = tex2DLayered<float4>(texture, rsx, rsy, blockIdx.z);
                accumulator[0] += read.x;
                accumulator[1] += read.y;
                accumulator[2] += read.z;
                accumulator[3] += read.w;
            }
            rsx += vx;
            rsy += vy;
        }


        
        #pragma unroll
        for (int b = 0; b < channels; b++) {
            output[(batch_id + b) * cfg.det_count * cfg.n_angles + angle_id * cfg.det_count + ray_id] =
                    accumulator[b] * n;
        }

    }
}

template<typename T>
void radon_forward_cuda(
        const T *x, const float *angles, T *y,
        TextureCache &tex_cache, const RaysCfg &cfg, const int batch_size, const int device
) {
//    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<true>, cudaFuncCachePreferL1));
//    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<false>, cudaFuncCachePreferL1));

    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = (batch_size % 4 == 0) ? 4 : 1;

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, cfg.height, cfg.width, 0, channels, precision});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(roundup_div(cfg.det_count, 16), roundup_div(cfg.n_angles, 16), batch_size / channels);

    if (cfg.is_fanbeam()) {
        if (channels == 1) {
            if (cfg.clip_to_circle) {
                radon_forward_kernel<false, 1, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else {
                radon_forward_kernel<false, 1, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            }
        } else {
            if (is_float) {
                if (cfg.clip_to_circle) {
                    radon_forward_kernel<false, 4, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                } else {
                    radon_forward_kernel<false, 4, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                }
            } else {
                if (cfg.clip_to_circle) {
                    radon_forward_kernel<false, 4, true> << < grid_dim, block_dim >> >
                                                                        ((__half *) y, tex->texture, angles, cfg);
                } else {
                    radon_forward_kernel<false, 4, false> << < grid_dim, block_dim >> >
                                                                         ((__half *) y, tex->texture, angles, cfg);
                }
            }
        }
    } else {
        if (channels == 1) {
            if (cfg.clip_to_circle) {
                radon_forward_kernel<true, 1, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else {
                radon_forward_kernel<true, 1, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            }
        } else {
            if (is_float) {
                if (cfg.clip_to_circle) {
                    radon_forward_kernel<true, 4, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                } else {
                    radon_forward_kernel<true, 4, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                }
            } else {
                if (cfg.clip_to_circle) {
                    radon_forward_kernel<true, 4, true> << < grid_dim, block_dim >> >
                                                                       ((__half *) y, tex->texture, angles, cfg);
                } else {
                    radon_forward_kernel<true, 4, false> << < grid_dim, block_dim >> >
                                                                        ((__half *) y, tex->texture, angles, cfg);
                }
            }
        }
    }
}

template void
radon_forward_cuda<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache, const RaysCfg &cfg,
                          const int batch_size, const int device);

template void radon_forward_cuda<unsigned short>(const unsigned short *x, const float *angles, unsigned short *y,
                                                 TextureCache &tex_cache, const RaysCfg &cfg,
                                                 const int batch_size, const int device);


template<int channels, typename T>
__global__ void
radon_forward_kernel_3d(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                     RaysCfg cfg) {
    // Calculate sensor coordinates in pixels
    const int pu = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int pv = blockIdx.z * blockDim.z + threadIdx.z;


    if (angle_id < cfg.n_angles && pu < cfg.det_count && pv < cfg.det_count_z) {
        float accumulator = 0.0f;

        // compute ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        float sx = 0.0f;
        float sy = -cfg.s_dist;
        // sz = initial_z + pitch * angle / (2*pi);
        float rsz = cfg.initial_z + cfg.pitch * angle * 0.1591549f;

        float ex = (pu - cfg.det_count * 0.5f + 0.5f) * cfg.det_spacing;
        float ey = cfg.d_dist;
        float rdz = (pv - cfg.det_count_z * 0.5f + 0.5f) * cfg.det_spacing_z;

        // rotate start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;

        rsx += cfg.width * 0.5f;
        rsy += cfg.height * 0.5f;

        // Clip ray to cube (with a little bit of padding) to reduce the number of memory reads
        constexpr float pad = 1.0f;
        const float alpha_x_m = (-pad - rsx)/rdx;
        const float alpha_x_p = (cfg.width + pad - rsx)/rdx;
        const float alpha_y_m = (-pad - rsy)/rdy;
        const float alpha_y_p = (cfg.height + pad - rsy)/rdy;
        const float alpha_z_m = (-pad - rsz)/rdz;
        const float alpha_z_p = (cfg.depth + pad - rsz)/rdz;

        const float alpha_s = max(min(alpha_x_p, alpha_x_m), max(min(alpha_y_p, alpha_y_m), min(alpha_z_p, alpha_z_m)));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), min(max(alpha_y_p, alpha_y_m), max(alpha_z_p, alpha_z_m)));

        if(alpha_s > alpha_e){
            output[angle_id * cfg.det_count_z * cfg.det_count + pv * cfg.det_count + pu] = 0.0f;
            return;
        }

        rsx += rdx*alpha_s;
        rsy += rdy*alpha_s;
        rsz += rdz*alpha_s;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);
        rdz *= (alpha_e - alpha_s);

        // accumulate loop
        const uint n_steps = __float2uint_ru( norm3df(rdx, rdy, rdz));
        const float vx = rdx / n_steps;
        const float vy = rdy / n_steps;
        const float vz = rdz / n_steps;
        const float n = norm3df(vx, vy, vz);

        for (uint j = 0; j <= n_steps; j++) { //changing j and n_steps to int makes everything way slower (WHY???)
            accumulator += tex3D<float>(texture, rsx, rsy, rsz);

            rsx += vx;
            rsy += vy;
            rsz += vz;
        }

        // output
        output[angle_id * cfg.det_count_z * cfg.det_count + pv * cfg.det_count + pu] = accumulator * n;
    }
}

void radon_forward_cuda_3d(
        const float *x, const float *angles, float *y,
        TextureCache &tex_cache, const RaysCfg &cfg, const int batch_size, const int device
){
    constexpr bool is_float = true;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = 1; //(batch_size % 4 == 0) ? 4 : 1;

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, 0, cfg.height, cfg.width, cfg.depth, channels, precision});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(
        roundup_div(cfg.det_count, block_dim.x),
        roundup_div(cfg.n_angles, block_dim.y),
        roundup_div(cfg.det_count_z, block_dim.z)
    );

    radon_forward_kernel_3d<1> <<< grid_dim, block_dim >>> (y, tex->texture, angles, cfg);
}