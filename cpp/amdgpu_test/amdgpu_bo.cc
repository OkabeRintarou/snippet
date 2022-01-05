#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <cstddef>
#include "amdgpu_bo.h"
#include "result.h"

using namespace amdgpu;

BufferObject::~BufferObject() {
    munmap();

    if (va_handle_ != nullptr) {
        amdgpu_bo_va_op(bo_handle_, 0, bo_size_, virtual_mc_base_address_, 0, AMDGPU_VA_OP_UNMAP);
        amdgpu_va_range_free(va_handle_);
        virtual_mc_base_address_ = 0;
        va_handle_ = nullptr;
    }

    if (bo_handle_ != nullptr) {
        amdgpu_bo_free(bo_handle_);
        bo_handle_ = nullptr;
    }
}

Result<BufferObject::Ptr, int> BufferObject::mmap() {
    if (!mapped_ptr_) {
        int r = amdgpu_va_range_alloc(device_handle_, amdgpu_gpu_va_range_general,
                    bo_size_, bo_align_, 0,
                    &virtual_mc_base_address_, &va_handle_, 0);
        if (r) {
            return make_err(r);
        }

        r = amdgpu_bo_va_op(bo_handle_, 0, bo_size_, virtual_mc_base_address_, 0, AMDGPU_VA_OP_MAP);
        if (r) {
            return make_err(r);
        }
        void *ptr = nullptr;
        r = amdgpu_bo_cpu_map(bo_handle_, &ptr);
        if (r) {
            return make_err(r);
        }
        mapped_ptr_.addr_ = ptr;
        mapped_ptr_.size_ = bo_size_;
    }
    return make_ok(mapped_ptr_);
}

void BufferObject::munmap() {
    if (mapped_ptr_) {
        amdgpu_bo_cpu_unmap(bo_handle_);
        mapped_ptr_.addr_ = nullptr;
        mapped_ptr_.size_ = 0;
    }
}
