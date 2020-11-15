#include <iostream>
#include <cuda_fp16.h>
#include <string>

#include "texture.h"
#include "utils.h"

TextureConfig::TextureConfig(int dv, int z, int y, int x, bool layered, int c, int p)
        : device(dv), depth(z), height(y), width(x), is_layered(layered), channels(c), precision(p) {}

bool TextureConfig::operator==(const TextureConfig &o) const {
    return this->device == o.device && this->width == o.width && this->height == o.height &&
           this->is_layered == o.is_layered && this->depth == o.depth && this->channels == o.channels &&
           this->precision == o.precision;
}

std::ostream &operator<<(std::ostream &os, TextureConfig const &m) {
    std::string precision = m.precision == PRECISION_FLOAT ? "float" : "half";

    return os << "(device: " << m.device << ", depth: " << m.depth << ", height: " << m.height << ", width: " << m.width
              << ", channels: " << m.channels << ", precision: " << precision << ", is layered: " << m.is_layered  << ")";
}

template<bool is_layered>
__global__ void
write_to_surface(const float *data, cudaSurfaceObject_t surface, const int width, const int height,
                 const int depth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        const int pitch = width * height * depth;
        const int offset = (z * height + y) * width + x;

        float4 tmp;
        tmp.x = data[0 * pitch + offset];
        tmp.y = data[1 * pitch + offset];
        tmp.z = data[2 * pitch + offset];
        tmp.w = data[3 * pitch + offset];

        if (is_layered) {
            surf2DLayeredwrite<float4>(tmp, surface, x * sizeof(float4), y, z);
        } else {
            surf3Dwrite<float4>(tmp, surface, x * sizeof(float4), y, z);
        }

//        if(x < 2 && y < 2 && z < 2){
//            printf("W %d %d %d = %f\n", x, y, z, tmp.x);
//            float v = surf2DLayeredread<float4>(surface, x * sizeof(float4), y, z, cudaBoundaryModeClamp).x;
//            printf("R %d %d %d = %f\n", x, y, z, v);
//        }
    }
}

template<bool is_layered>
__global__ void
write_half_to_surface(const __half *data, cudaSurfaceObject_t surface, const int width, const int height,
                      const int depth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        const int pitch = width * height * depth;
        const int offset = (z * height + y) * width + x;

        __half tmp[4];
        for (int i = 0; i < 4; i++) tmp[i] = __float2half(data[i * pitch + offset]);

        if (is_layered) {
            surf2DLayeredwrite<float2>(*(float2 *) tmp, surface, x * sizeof(float2), y, z);
        } else {
            surf3Dwrite<float2>(*(float2 *) tmp, surface, x * sizeof(float2), y, z);
        }
    }
}


cudaChannelFormatDesc get_channel_desc(int channels, int precision) {
    if (precision == PRECISION_FLOAT) {
        if (channels == 1) {
            return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        }
        if (channels == 4) {
            return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        }
    }
    if (precision == PRECISION_HALF && channels == 4) {
        return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    }
    std::cerr << "[TORCH RADON] ERROR unsupported number of channels and precision (channels:" << channels
              << ", precision: " << precision << ")" << std::endl;
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
}

Texture::Texture(TextureConfig c) :cfg(c) {
    checkCudaErrors(cudaSetDevice(this->cfg.device));

#ifdef VERBOSE
    std::cout << "[TORCH RADON] Allocating Texture " << this->cfg << std::endl;
#endif

    // Allocate CUDA array
    cudaChannelFormatDesc channelDesc = get_channel_desc(cfg.channels, cfg.precision);
    const cudaExtent extent = make_cudaExtent(cfg.width, cfg.height, cfg.depth);
    auto allocation_type = cfg.is_layered ? cudaArrayLayered : cudaArrayDefault;
    checkCudaErrors(cudaMalloc3DArray(&array, &channelDesc, extent, allocation_type));

    // Create resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));

    // Create surface object
    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
}

void Texture::put(const float *data) {
    if (this->cfg.precision == PRECISION_HALF) {
        std::cerr << "[TORCH RADON] ERROR putting half precision data into a float texture" << std::endl;
    }

    checkCudaErrors(cudaSetDevice(this->cfg.device));

    if (cfg.channels == 1) {
        // if using a single channel use cudaMemcpy to copy data into array
        cudaMemcpy3DParms myparms = {0};
        myparms.srcPos = make_cudaPos(0, 0, 0);
        myparms.dstPos = make_cudaPos(0, 0, 0);
        myparms.srcPtr = make_cudaPitchedPtr((void *) data, cfg.width * sizeof(float), cfg.width, cfg.height);
        myparms.dstArray = this->array;

        myparms.extent = make_cudaExtent(cfg.width, cfg.height, cfg.depth);

        myparms.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&myparms));
    } else {
        // else if using multiple channels use custom kernel to copy the data
        dim3 grid_dim(roundup_div(cfg.width, 16), roundup_div(cfg.height, 16), cfg.depth);
        if (cfg.is_layered) {
            write_to_surface<true> << < grid_dim, dim3(16, 16) >> >
                                                  (data, this->surface, cfg.width, cfg.height, cfg.depth);
        } else {
            write_to_surface<false> << < grid_dim, dim3(16, 16) >> >
                                                   (data, this->surface, cfg.width, cfg.height, cfg.depth);
        }
    }
}

void Texture::put(const unsigned short *data) {
    if (this->cfg.precision == PRECISION_FLOAT) {
        std::cerr << "[TORCH RADON] ERROR putting single precision data into a half precision texture" << std::endl;
    }

    checkCudaErrors(cudaSetDevice(this->cfg.device));

    dim3 grid_dim(roundup_div(cfg.width, 16), roundup_div(cfg.height, 16), cfg.depth);
    if (cfg.is_layered) {
        write_half_to_surface<true> << < grid_dim, dim3(16, 16) >> >
                                                   ((__half *) data, this->surface, cfg.width, cfg.height, cfg.depth);
    } else {
        write_half_to_surface<false> << < grid_dim, dim3(16, 16) >> >
                                                    ((__half *) data, this->surface, cfg.width, cfg.height, cfg.depth);
    }
}

bool Texture::matches(TextureConfig &c) {
    return c == this->cfg;
}

Texture::~Texture() {
#ifdef VERBOSE
    std::cout << "[TORCH RADON] Freeing Texture " << this->cfg << std::endl;
#endif
    if (this->array != nullptr) {
        checkCudaErrors(cudaSetDevice(this->cfg.device));
        checkCudaErrors(cudaDestroyTextureObject(this->texture));
        checkCudaErrors(cudaDestroySurfaceObject(this->surface));
        checkCudaErrors(cudaFreeArray(this->array));
        this->array = nullptr;
    }
}