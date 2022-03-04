#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <amdgpu.h>
#include <xf86drm.h>
#include "amdgpu_device.h"

using namespace amdgpu;

bool Devices::unload(Devices &devices) {
    // TODO: unload devices
    return true;
}

bool Devices::load(Devices &devices, bool open_render_node) {

	const int MAX_CARDS_SUPPORTED = 128;
	drmDevicePtr drm_devices[MAX_CARDS_SUPPORTED];

	int drm_count = drmGetDevices2(0, drm_devices, MAX_CARDS_SUPPORTED);
	if (drm_count < 0) {
		return false;
	}	

	for (int i = 0; i < drm_count; i++) {
		// skip if this is not PCI device
		if (drm_devices[i]->bustype != DRM_BUS_PCI) {
			continue;
		}
		// skip if this not AMD GPU vender ID
		if (drm_devices[i]->deviceinfo.pci->vendor_id != 0x1002) {
			continue;
		}

		const int drm_mode = open_render_node ? DRM_NODE_RENDER : DRM_NODE_PRIMARY;
		int fd = -1;
		if (drm_devices[i]->available_nodes & 1 << drm_mode) {
			fd = open(drm_devices[i]->nodes[drm_mode], O_RDWR | O_CLOEXEC);
		} 

		if (fd < 0) {
			continue;
		}

		drmVersionPtr version = drmGetVersion(fd);
		if (version == nullptr) {
			close(fd);
			continue;
		}


		if (strcmp(version->name, "amdgpu") != 0) {
			// skip if this is not AMDGPU driver
			drmFreeVersion(version);
			close(fd);
			continue;
		}

        int r;
        uint32_t major_version, minor_version;
        amdgpu_device_handle device_handle;
        r = amdgpu_device_initialize(fd, &major_version, &minor_version, &device_handle);
        if (r) {
            // TODO: add error log
            drmFreeVersion(version);
            close(fd);
        }

        amdgpu_gpu_info gpu_info{0};
        r = amdgpu_query_gpu_info(device_handle, &gpu_info);
        if (r) {
            // TODO: add error log
            drmFreeVersion(version);
            close(fd);
        }

        devices.emplace_back(fd, device_handle);        
        Device &d = devices.back();
        d.device_info_.major_version = major_version;
        d.device_info_.minor_version = minor_version;
        d.device_info_.gpu_info = gpu_info;

		drmFreeVersion(version);
	}

	drmFreeDevices(drm_devices, drm_count);

	return !devices.empty();
}

Devices::~Devices() {
	for (size_t i = 0; i < size(); i++) {
        const Device& d = (*this)[i];
		close(d.fd());
	}
}

Result<BufferObject, int> Device::alloc_bo(amdgpu_bo_alloc_request req) {
    amdgpu_bo_handle bo_handle;
    int r = amdgpu_bo_alloc(handle_, &req, &bo_handle);
    if (r) {
        return make_err(r);
    }
    
    return make_ok(BufferObject(handle_, bo_handle, req.alloc_size, req.phys_alignment));
}

Result<BufferObject, int> Device::alloc_bo(uint64_t alloc_size, uint64_t alignment, uint32_t preferred_heap, uint64_t flags) {
    amdgpu_bo_alloc_request req{0};
    req.alloc_size = alloc_size;
    req.phys_alignment = alignment;
    req.preferred_heap = preferred_heap;
    req.flags = flags;
    
    return alloc_bo(req);
}

Result<Context, int> Device::alloc_context() {
    amdgpu_context_handle context_handle;

    int r = amdgpu_cs_ctx_create(handle_, &context_handle);
    if (r) {
        return make_err(r);
    }
    return make_ok(Context{context_handle});
}
