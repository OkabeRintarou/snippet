#include <amdgpu.h>
#include <drm.h>
#include <amdgpu_drm.h>
#include "basic_test.h"

bool BasicTest::init() {
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
	return true;
}

TEST_F(BasicTest, QueryInfoTest) {
	amdgpu_gpu_info gpu_info{0};
	uint32_t version, feature;
	
	int r = amdgpu_query_gpu_info(device_handle_, &gpu_info);
	ASSERT_EQ(r, 0);

	r = amdgpu_query_firmware_version(device_handle_, AMDGPU_INFO_FW_VCE, 0, 0, &version, &feature);
	ASSERT_EQ(r, 0);
}
