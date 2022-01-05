#pragma once

#include <gtest/gtest.h>
#include "amdgpu_device.h"

class BasicTest : public testing::Test {
protected:
	virtual void SetUp() override {
		ASSERT_TRUE(init());
	}

	virtual void TearDown() override {
        ASSERT_TRUE(fini());
	}

	amdgpu::Devices devices_;
private:
	bool init();
    bool fini();
};
