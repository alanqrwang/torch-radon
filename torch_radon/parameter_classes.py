from torch_radon_cuda import VolumeCfg, ProjectionCfg, ExecCfg

"""
Classes that wrap their C++ counterparts offering easy to use APIs 
"""


class Volume:
    def __init__(self, depth, height, width, dz, dy, dx, is_3d):
        self.cfg = VolumeCfg(
            depth, height, width,
            dz, dy, dx,
            is_3d
        )

    @staticmethod
    def create_2d(height, width=-1, dy=0.0, dx=0.0):
        if width <= 0:
            width = height

        return Volume(
            0, height, width,
            0.0, dy, dx,
            False
        )

    @staticmethod
    def create_3d(depth, height=-1, width=-1, dz=0.0, dy=0.0, dx=0.0):
        if height <= 0:
            height = depth

        if width <= 0:
            width = height

        return Volume(
            depth, height, width,
            dz, dy, dx,
            True
        )

    def max_dim(self):
        return max(self.cfg.depth, self.cfg.height, self.cfg.width)

    def num_dimensions(self):
        return 3 if self.cfg.is_3d else 2


class Projection:
    PARALLEL = 0
    FANBEAM = 1
    CONE_FLAT = 2

    def __init__(self, cfg: ProjectionCfg):
        self.cfg = cfg

    @staticmethod
    def parallel_beam(det_count, det_spacing=1.0):
        return Projection(ProjectionCfg(det_count, det_spacing))

    @staticmethod
    def fanbeam(src_dist, det_dist, det_count, det_spacing=1.0):
        return Projection(ProjectionCfg(
            det_count, det_spacing,
            0, 1.0,
            src_dist, det_dist,
            0.0, 0.0,
            Projection.FANBEAM
        ))

    def is_2d(self):
        return self.cfg.is_2d()
