#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <amdgpu_drm.h>
#include <xf86drm.h>
#include "amdgpu_test.h"

#define ALIGN(v, a) (((v) + (a) - 1) & ~((a) - 1))

int open_device(bool open_render_mode) {
    const int MAX_CARDS_SUPPORTED = 128;
    drmDevicePtr drm_devices[MAX_CARDS_SUPPORTED];

    int drm_cnt = drmGetDevices2(0, drm_devices, MAX_CARDS_SUPPORTED);
    if (drm_cnt < 0)
        return -1;

    for (int i = 0; i < drm_cnt; i++) {
        // skip if this is not PCI device
        if (drm_devices[i]->bustype != DRM_BUS_PCI)
            continue;
        // skip if this is not AMD GPU vender ID
        if (drm_devices[i]->deviceinfo.pci->vendor_id != 0x1002)
            continue;

        const int drm_mode = open_render_mode ? DRM_NODE_RENDER : DRM_NODE_PRIMARY;
        int fd = -1;
        if (drm_devices[i]->available_nodes & (1 << drm_mode))
            fd = open(drm_devices[i]->nodes[drm_mode], O_RDWR | O_CLOEXEC);
        if (fd < 0)
            continue;
        drmVersionPtr version = drmGetVersion(fd);
        if (version == nullptr) {
            close(fd);
            continue;
        }
        if (strcmp(version->name, "amdgpu") != 0) {
            drmFreeVersion(version);
            close(fd);
            continue;
        }
        return fd;
    }
    return -1;
}

int amdgpu_bo_alloc_and_map_raw(amdgpu_device_handle dev, uint64_t size,
                                uint64_t alignment, unsigned heap, uint64_t alloc_flags,
                                uint64_t mapping_flags, amdgpu_bo_handle *bo, void **cpu,
                                uint64_t *mc_address,
                                amdgpu_va_handle *va_handle) {
    amdgpu_bo_alloc_request request {};
    amdgpu_bo_handle buf_handle;
    amdgpu_va_handle handle;
    uint64_t vmc_addr;
    int r;

    request.alloc_size = size;
    request.phys_alignment = alignment;
    request.preferred_heap = heap;
    request.flags = alloc_flags;

    r = amdgpu_bo_alloc(dev, &request, &buf_handle);
    if (r)
        return r;

    r = amdgpu_va_range_alloc(dev,
                              amdgpu_gpu_va_range_general,
                              size, alignment, 0, &vmc_addr,
                              &handle, 0);
    if (r)
        goto error_va_alloc;

    r = amdgpu_bo_va_op_raw(dev, buf_handle, 0, ALIGN(size, getpagesize()), vmc_addr,
                            AMDGPU_VM_PAGE_READABLE |
                            AMDGPU_VM_PAGE_WRITEABLE |
                            AMDGPU_VM_PAGE_EXECUTABLE |
                            mapping_flags,
                            AMDGPU_VA_OP_MAP);
    if (r)
        goto error_va_map;

    r = amdgpu_bo_cpu_map(buf_handle, cpu);
    if (r)
        goto error_cpu_map;

    *bo = buf_handle;
    *mc_address = vmc_addr;
    *va_handle = handle;
    return 0;

error_cpu_map:
    amdgpu_bo_cpu_unmap(buf_handle);
error_va_map:
    amdgpu_bo_va_op(buf_handle, 0, size, vmc_addr, 0, AMDGPU_VA_OP_UNMAP);
error_va_alloc:
    amdgpu_bo_free(buf_handle);

    return r;
}
