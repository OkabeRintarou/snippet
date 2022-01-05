#include <amdgpu.h>
#include <drm.h>
#include <amdgpu_drm.h>
#include "amdgpu_device.h"
#include "basic_test.h"

using namespace amdgpu;

bool BasicTest::init() {
    return Devices::load(devices_);
}

bool BasicTest::fini() {
    return Devices::unload(devices_);
}

TEST_F(BasicTest, QueryInfoTest) {
	amdgpu_gpu_info gpu_info{0};
	uint32_t version, feature;
	
    Device &device = devices_[0];
	int r = amdgpu_query_gpu_info(device.raw_handle(), &gpu_info);
	ASSERT_EQ(r, 0);

	r = amdgpu_query_firmware_version(device.raw_handle(), AMDGPU_INFO_FW_VCE, 0, 0, &version, &feature);
	ASSERT_EQ(r, 0);
}

