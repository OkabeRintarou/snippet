#pragma once

#include <gtest/gtest.h>
#include <amdgpu.h>
#include "amdgpu_device.h"

class VMTest : public testing::Test {
protected:
    void SetUp() override;
protected:
    amdgpu::Device dev_;
};
