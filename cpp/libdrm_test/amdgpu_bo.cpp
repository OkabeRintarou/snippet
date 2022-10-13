#include "amdgpu_test.h"
#include "amdgpu_bo.h"

namespace amdgpu {

BufferObject::~BufferObject() {
    if (bo_handle_) {
        amdgpu_bo_unmap_and_free(bo_handle_, va_handle_, mc_address_, size_);
    }
}

}
