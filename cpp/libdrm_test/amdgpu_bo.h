#pragma once

#include <amdgpu.h>

namespace amdgpu {

class BufferObject {
public:
    friend class Device;
    ~BufferObject();
    uint64_t size() const { return size_; }
    uint64_t gpu_address() const { return mc_address_; }
    void *cpu_address() const { return cpu_address_; }
    amdgpu_bo_handle handle() const { return bo_handle_; }

    bool is_valid() const { return bo_handle_ != nullptr; }
private:
    BufferObject() = default;
    BufferObject(uint64_t size, uint64_t mc_addr, void *cpu_addr, amdgpu_bo_handle bo_handle, amdgpu_va_handle va_handle)
        : size_(size),
          mc_address_(mc_addr),
          cpu_address_(cpu_addr),
          bo_handle_(bo_handle),
          va_handle_(va_handle) {}

    uint64_t size_ = 0;
    uint64_t mc_address_ = 0;
    void *cpu_address_ = nullptr;
    amdgpu_bo_handle bo_handle_ = nullptr;
    amdgpu_va_handle va_handle_ = nullptr;
};

}