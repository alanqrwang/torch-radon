#include "cache.h"

DeviceSizeKey::DeviceSizeKey(int dv, int b, int w, int h, int d, int c, int p) : device(dv), batch(b), width(w), height(h),
                                                                         depth(d), channels(c), precision(p) {}

bool DeviceSizeKey::operator==(const DeviceSizeKey &o) const {
    return this->device == o.device && this->batch == o.batch && this->width == o.width && this->height == o.height &&
           this->depth == o.depth && this->channels == o.channels && this->precision == o.precision;
}

bool DeviceSizeKey::is_3d() const{
    return this->depth > 0;
}

int DeviceSizeKey::z_size() const{
    return this->is_3d() ? this->depth : this->batch;
}

std::ostream &operator<<(std::ostream &os, DeviceSizeKey const &m) {
    return os << "(device: " << m.device << ", batch: " << m.batch << ", width: " << m.width << ", height: " << m.height
             << ", depth: " << m.depth << ", channels: " << m.channels << ", precision: " << m.precision << ")";
}