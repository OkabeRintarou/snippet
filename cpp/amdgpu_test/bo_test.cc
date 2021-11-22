#include <amdgpu.h>
#include <drm.h>
#include <amdgpu_drm.h>
#include "bo_test.h"

static const int BUFFER_SIZE = 4096;
static const int BUFFER_ALIGN = 4096;

bool BoTest::init() {
	if (!open_devices(devices_, false)) {
		return false;
	}
	
	int r = amdgpu_device_initialize(
			devices_[0], &major_version_,
			&minor_version_, &device_handle_);
	if (r) {
		return false;
	}

	amdgpu_gpu_info gpu_info{0};
	r = amdgpu_query_gpu_info(device_handle_, &gpu_info);
	if (r) {
		return false;
	}
	family_id_ = gpu_info.family_id;
    
    amdgpu_bo_alloc_request req{0};
    req.alloc_size = BUFFER_SIZE;
    req.phys_alignment = BUFFER_ALIGN;
    req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;

    amdgpu_bo_handle buf_handle;
    r = amdgpu_bo_alloc(device_handle_, &req, &buf_handle);
    if (r) {
        return false;
    }

    uint64_t va = 0;
    r = amdgpu_va_range_alloc(device_handle_, amdgpu_gpu_va_range_general, BUFFER_SIZE, BUFFER_ALIGN, 0, &va, &va_handle_, 0);
    if (r) {
        return false;
    }
    
    r = amdgpu_bo_va_op(buf_handle, 0, BUFFER_SIZE, va, 0, AMDGPU_VA_OP_MAP);
    if (r) {
        return false;
    }

    buffer_handle_ = buf_handle;
    virtual_mc_base_address_ = va;
    
	return true;
}

TEST_F(BoTest, MapUnmapTest) {
    uint32_t *ptr;
    int r;

    r = amdgpu_bo_cpu_map(buffer_handle_, (void**)&ptr);
    ASSERT_EQ(r, 0);
    ASSERT_NE(ptr, nullptr);

    for (int i = 0, e = BUFFER_SIZE / sizeof(uint32_t); i < e; i++) {
        ptr[i] = 0xdeadbeaf;
    }
    r = amdgpu_bo_cpu_unmap(buffer_handle_);
    ASSERT_EQ(r, 0);
}
