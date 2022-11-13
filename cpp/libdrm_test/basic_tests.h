#pragma once

#include <gtest/gtest.h>
#include <amdgpu.h>
#include <vector>
#include "amdgpu_device.h"

class PM4WriteDataPacket;

class BasicTest : public testing::Test {
protected:
    void SetUp() override;
protected:
    void exec_cs_helper_raw(amdgpu::Context &ctx,
                            unsigned ip,
                            unsigned ring_id,
                            const void *pm4_packet,
                            unsigned packet_size,
                            int res_cnt, amdgpu_bo_handle *resources,
                            bool secure);
    void command_submission_write_linear_helper_with_secure(unsigned ip_type, bool secure);
    void command_submission_copy_linear_helper(unsigned ip_type);
protected:
    amdgpu::Device dev_;
};