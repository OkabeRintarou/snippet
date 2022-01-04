#pragma once

#include <gtest/gtest.h>
#include "amdgpu_device.h"

class BoTest : public testing::Test {
protected:
	virtual void SetUp() override {
		ASSERT_TRUE(init());
	}

	virtual void TearDown() override {
        ASSERT_TRUE(fini());
		amdgpu_device_deinitialize(device_handle_);	
	}

	amdgpu::Devices devices_;

	amdgpu_device_handle device_handle_;
	uint32_t major_version_;
	uint32_t minor_version_;
	uint32_t family_id_;
	uint32_t chip_id_;
	uint32_t chip_rev_;

    amdgpu_bo_handle buffer_handle_;
    uint64_t virtual_mc_base_address_;
    amdgpu_va_handle va_handle_;
private:
	bool init();
    bool fini();
};
