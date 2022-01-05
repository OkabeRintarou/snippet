#pragma once

#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <vector>
#include "amdgpu_bo.h"
#include "result.h"

namespace amdgpu {

class Devices;

struct DeviceInfo {
    uint32_t major_version;
    uint32_t minor_version;
    amdgpu_gpu_info gpu_info;
};

class Device {
public:
    Device(int fd, amdgpu_device_handle handle) : fd_(fd), handle_(handle) {}

    const int fd() const { return fd_; }
    amdgpu_device_handle raw_handle() const { return handle_; }

    Result<BufferObject, int> alloc_bo(amdgpu_bo_alloc_request req);
    Result<BufferObject, int> alloc_bo(uint64_t alloc_size, uint64_t alignment = 0, uint32_t preferred_heap = AMDGPU_GEM_DOMAIN_GTT, uint64_t flags = 0);

private:
    Device() = default;
    friend class Devices;

    int fd_ = -1;
    amdgpu_device_handle handle_;
};

class Devices : private std::vector<Device> {
private:
	using self = std::vector<Device>;
public:
	using self::empty;
	using self::size;
    using self::front;
    using self::back;
	using self::operator[];

    static bool load(Devices &devices, bool open_render_node = false);
    static bool unload(Devices &devices);

    Devices() = default;
	Devices(const Devices&) = delete;
	Devices(Devices &&) = delete;
	Devices& operator=(const Devices&) = delete;
	Devices& operator=(Devices &&) = delete;

	~Devices();
};


}
