#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <xf86drm.h>

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

