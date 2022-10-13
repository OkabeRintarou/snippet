#pragma once

#include <amdgpu.h>
#include <memory>
#include "amdgpu_bo.h"

namespace amdgpu {

class Device {
public:
    Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool is_valid() const { return fd_ != -1 && dev_ != nullptr; }
    amdgpu_device_handle handle() const { return dev_; }

    amdgpu::BufferObject alloc_bo(
        uint64_t size, uint64_t align, unsigned heap, uint64_t flags = 0);
private:
    int fd_ = -1;
    amdgpu_device_handle  dev_ = nullptr;
    uint32_t major_version_;
    uint32_t minor_version_;
};
}
