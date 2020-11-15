#include "parameter_classes.h"

VolumeCfg::VolumeCfg(int d, int h, int w, float _dz, float _dy, float _dx, bool ddd)
        : depth(d), height(h), width(w), dz(_dz), dy(_dy), dx(_dx), is_3d(ddd) {}

ProjectionCfg::ProjectionCfg(int dc_u, float ds_u, int dc_v, float ds_v, int na, bool ctc, float sd, float dd,
                  float pi, float iz, int pt)
            : det_count_u(dc_u), det_spacing_u(ds_u), det_count_v(dc_v),
              det_spacing_v(ds_v), n_angles(na), clip_to_circle(ctc), s_dist(sd), d_dist(dd), pitch(pi), initial_z(iz),
              projection_type(pt) {}


ExecCfg::ExecCfg(int x, int y, int z, int ch)
        :block_dim(x, y, z), channels(ch) {}

dim3 ExecCfg::get_grid_size(int x, int y, int z) const{
    return dim3(roundup_div(x, block_dim.x), roundup_div(y, block_dim.y), roundup_div(z, block_dim.z));
}

int ExecCfg::get_channels(int batch_size) const{
    return (batch_size % 4 == 0) ? this->channels : 1;
}