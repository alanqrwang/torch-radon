#include <cuda.h>
#include "defines.h"
#include "utils.h"

#ifndef TORCH_RADON_PARAMETER_CLASSES_H
#define TORCH_RADON_PARAMETER_CLASSES_H

class VolumeCfg {
public:
    // dimensions of the measured volume
    int depth;
    int height;
    int width;

    // position delta on each axis
    float dz;
    float dy;
    float dx;

    bool is_3d;

    VolumeCfg(int d, int h, int w, float _dz, float _dy, float _dx, bool ddd);

//    __device__ __inline__ float get_bound() const {
//        // return the maximum distance from the origin of a point contained in the volume
//
//        // assume it is not 3D
//        return hypot(abs(dx) + width * 0.5f, abs(dy) + height * 0.5f);
//    }
//
//    __device__ __inline__ float min_x() const { return dx - 0.5f * width; }
//
//    __device__ __inline__ float max_x() const { return dx + 0.5f * width; }
//
//    __device__ __inline__ float min_y() const { return dy - 0.5f * height; }
//
//    __device__ __inline__ float max_y() const { return dy + 0.5f * height; }

};

class ProjectionCfg {
public:
    // number of pixels of the detector and spacing
    int det_count_u;
    float det_spacing_u;
    int det_count_v;
    float det_spacing_v;

    int n_angles;
    bool clip_to_circle;

    // source and detector distances (for fanbeam and coneflat)
    float s_dist = 0.0;
    float d_dist = 0.0;

    // pitch = variation in z after a full rotation (for coneflat)
    float pitch;
    float initial_z;

    int projection_type;

    ProjectionCfg(int dc_u, float ds_u, int dc_v, float ds_v, int na, bool ctc, float sd, float dd,
                  float pi, float iz, int pt);

//    __device__ __inline__ float det_pixel_pos_u(int p) const {
//        //get the position of the detector pixel p on the u axis
//        return (p - det_count_u * 0.5f + 0.5f) * det_spacing_u;
//    }
};

class ExecCfg {
public:
    dim3 block_dim;

    int channels;

    ExecCfg(int x, int y, int z, int ch);

    dim3 get_grid_size(int x, int y = 1, int z = 1) const;

    int get_channels(int batch_size) const;
};

#endif
