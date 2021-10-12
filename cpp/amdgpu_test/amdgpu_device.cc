#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <amdgpu.h>
#include <xf86drm.h>
#include "amdgpu_device.h"

namespace amdgpu {

bool open_devices(Devices &devices, bool open_render_node) {

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

		devices.add(fd);

		drmFreeVersion(version);
	}

	drmFreeDevices(drm_devices, drm_count);

	return !devices.empty();
}

Devices::~Devices() {
	for (size_t i = 0; i < size(); i++) {
		close(this->operator[](i));
	}
}

} // namespace amdgpu
