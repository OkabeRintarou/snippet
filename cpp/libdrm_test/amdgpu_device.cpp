#include <unistd.h>
#include "amdgpu_test.h"
#include "amdgpu_device.h"

namespace amdgpu {

Device::Device() {
   fd_ = open_device();
   if (fd_ < 0)
       return;

   int r = amdgpu_device_initialize(fd_, &major_version_, &minor_version_, &dev_);
   if (r) {
       close(fd_);
       fd_ = -1;
       dev_ = nullptr;
   }
}

Device::~Device() {
    if (dev_)
        amdgpu_device_deinitialize(dev_);
    if (fd_ >= 0)
        close(fd_);
}

bool Device::alloc(const amdgpu_bo_alloc_request &req, BufferObject &bo) {
    uint64_t mc_addr;
    void *cpu;
    amdgpu_bo_handle bo_handle;
    amdgpu_va_handle va_handle;

    int r = amdgpu_bo_alloc_and_map(dev_,
                                    req.alloc_size,
                                    req.phys_alignment,
                                    req.preferred_heap,
                                    req.flags,
                                    &bo_handle,
                                    &cpu,
                                    &mc_addr,
                                    &va_handle);
    bo.code = r;
    if (r != 0) {
        return false;
    }
    bo.size_ = req.alloc_size;
    bo.mc_address_ = mc_addr;
    bo.cpu_address_ = cpu;
    bo.bo_handle_ = bo_handle;
    bo.va_handle_ = va_handle;

    return true;
}

bool Device::alloc(Context &ctx) {
    amdgpu_context_handle ctx_handle = nullptr;
    int r;

    r = amdgpu_cs_ctx_create(dev_, &ctx_handle);
    ctx.code = r;
    if (r != 0) {
        return false;
    }
    ctx.ctx_handle_ = ctx_handle;
    ctx.device_ = this;

    return true;
}

Context::~Context() {
    if (ctx_handle_ != nullptr) {
        amdgpu_cs_ctx_free(ctx_handle_);
    }
    device_ = nullptr;
}

BufferObject::~BufferObject() {
    if (bo_handle_ != nullptr) {
        amdgpu_bo_unmap_and_free(bo_handle_, va_handle_, mc_address_, size_);
    }
}

}
