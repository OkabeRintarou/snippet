#pragma once

#include <amdgpu.h>
#include <amdgpu_drm.h>

int open_device(bool open_render_mode = true);

int amdgpu_bo_alloc_and_map_raw(amdgpu_device_handle dev, uint64_t size,
                                uint64_t alignment, unsigned heap, uint64_t alloc_flags,
                                uint64_t mapping_flags, amdgpu_bo_handle *bo, void **cpu,
                                uint64_t *mc_address,
                                amdgpu_va_handle *va_handle);

inline int amdgpu_bo_alloc_and_map(amdgpu_device_handle dev, uint64_t size,
                                   uint64_t alignment, unsigned heap, uint64_t alloc_flags,
                                   amdgpu_bo_handle *bo, void **cpu, uint64_t *mc_address,
                                   amdgpu_va_handle *va_handle) {
    return amdgpu_bo_alloc_and_map_raw(dev, size, alignment, heap, alloc_flags,
                                       0, bo, cpu, mc_address, va_handle);
}

inline int amdgpu_bo_unmap_and_free(amdgpu_bo_handle bo, amdgpu_va_handle va_handle, uint64_t mc_addr, uint64_t size) {
    amdgpu_bo_cpu_unmap(bo);
    amdgpu_bo_va_op(bo, 0, size, mc_addr, 0, AMDGPU_VA_OP_UNMAP);
    amdgpu_va_range_free(va_handle);
    amdgpu_bo_free(bo);
    return 0;
}

inline int amdgpu_get_bo_list(amdgpu_device_handle dev,
                              amdgpu_bo_handle bo1, amdgpu_bo_handle bo2,
                              amdgpu_bo_list_handle *list) {
    amdgpu_bo_handle resources[] = {bo1, bo2};
    return amdgpu_bo_list_create(dev, bo2 ? 2 : 1, resources, nullptr, list);
}

inline bool asic_is_gfx_pipe_removed(uint32_t family_id, uint32_t chip_id, uint32_t chip_resv) {
    if (family_id != AMDGPU_FAMILY_AI)
        return false;

    switch (chip_id - chip_resv) {
        case 0x32:
        case 0x3c:
            return true;
        default:
            return false;
    }
}