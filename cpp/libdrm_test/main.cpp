#include <unistd.h>
#include <fcntl.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <amdgpu.h>
#include <xf86drm.h>
#include "amdgpu_test.h"

static int open_device(bool open_render_mode = true) {
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

int main() {
    int fd = open_device();
    if (fd < 0) {
        fprintf(stderr, "fail to open amdgpu device");
        return -1;
    }
    amdgpu_device_handle dev_handle = nullptr;
    uint32_t major_version, minor_version;
    int r;

    r = amdgpu_device_initialize(fd, &major_version, &minor_version, &dev_handle);
    if (r < 0) {
        fprintf(stderr, "fail to initialize amdgpu device: %d\n", r);
        close(fd);
        return -1;
    }

    struct amdgpu_gpu_info gpu_info {};
    r = amdgpu_query_gpu_info(dev_handle, &gpu_info);
    assert(r == 0);

    amdgpu_context_handle context_handle = nullptr;
    r = amdgpu_cs_ctx_create(dev_handle, &context_handle);
    assert(r == 0);

    amdgpu_device_deinitialize(dev_handle);

    close(fd);
    return 0;
}
