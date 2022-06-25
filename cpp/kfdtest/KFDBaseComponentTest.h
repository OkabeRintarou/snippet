#pragma once

#include <hsakmt.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>

class KFDBaseComponentTest : public testing::Test {
public:
    static constexpr unsigned MAX_RENDER_NODES = 64;

    struct {
        int fd = -1;
        uint32_t major_version = 0u;
        uint32_t minor_version = 0;
        amdgpu_device_handle device_handle = nullptr;
        uint32_t bdf = 0u;
    } m_RenderNodes[MAX_RENDER_NODES];
protected:
    HsaVersionInfo m_VersionInfo;
    HsaSystemProperties m_SystemProperties;
    unsigned int m_FamilyId;
    unsigned int m_numCpQueues;
    unsigned int m_numSdmaEngines;
    unsigned int m_numSdmaXgmiEngines;
    HsaMemFlags m_MemoryFlags;

    void SetUp() override;
    void TearDown() override;
};

extern KFDBaseComponentTest *g_baseTest;