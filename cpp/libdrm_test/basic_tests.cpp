#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <vector>
#include "basic_tests.h"
#include "pm4.h"

using namespace amdgpu;
using namespace std;

void BasicTest::SetUp() {
    ASSERT_TRUE(dev_.is_valid());
}

void BasicTest::command_submission_write_linear_helper_with_secure(unsigned int ip_type, bool secure) {
    const int sdma_write_length = 128;
    vector<uint32_t> write_data(sdma_write_length, 0xdeadbeaf);
    drm_amdgpu_info_hw_ip hw_ip_info{};
    int r, loop, i;
    uint32_t ring_id;
    uint64_t gtt_flags[2] = {0, AMDGPU_GEM_CREATE_CPU_GTT_USWC};
    vector<amdgpu_bo_handle> resources;

    r = amdgpu_query_hw_ip_info(dev_.handle(), ip_type, 0, &hw_ip_info);
    ASSERT_EQ(r, 0);

    for (i = 0; secure && (i < 2); i++)
        gtt_flags[i] |= AMDGPU_GEM_CREATE_ENCRYPTED;

    Context ctx;
    ASSERT_TRUE(dev_.alloc(ctx));

    for (ring_id = 0; (1 << ring_id) & hw_ip_info.available_rings; ++ring_id) {
        loop = 0;
        resources.clear();

        while (loop < 2) {
            amdgpu_bo_alloc_request req{};
            BufferObject bo;

            req.alloc_size = sdma_write_length * sizeof(uint32_t);
            req.phys_alignment = 4096;
            req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;
            req.flags = gtt_flags[loop];

            ASSERT_TRUE(dev_.alloc(req, bo));

            memset((void*)bo.cpu_address(), 0, sdma_write_length * sizeof(uint32_t));

            resources.push_back(bo.handle());

            if (ip_type == AMDGPU_HW_IP_GFX || ip_type == AMDGPU_HW_IP_COMPUTE) {
                // prepare pm4 pm4
                PM4WriteDataPacket::Config conf{};
                //conf.engine_sel = MecWriteData::EngineSel::CE;
                conf.write_confirm = MecWriteData::WriteConfirm::YES;
                conf.dst_sel = MecWriteData::DestSel::MEMORY_ASYNC;
                PM4WriteDataPacket pm4(conf, bo.gpu_address(), write_data.data(), write_data.size());
                // execute command
                exec_cs_helper_raw(ctx, ip_type, ring_id,
                                   pm4.get_packet(), pm4.size_in_bytes(),
                                   (int)resources.size(), resources.data(),
                                   secure);
                // verify if SDMA test result meets with expected
                i = 0;
                if (!secure) {
                    while (i < sdma_write_length) {
                        auto cpu_addr = reinterpret_cast<uint32_t *>(bo.cpu_address());
                        ASSERT_EQ(cpu_addr[i], 0xdeadbeaf);
                        ++i;
                    }
                }
            }

            ++loop;
        }
    }
}

void BasicTest::exec_cs_helper_raw(Context &ctx,
                                   unsigned ip, unsigned ring_id,
                                   const void *pm4_packet,
                                   unsigned packet_size,
                                   int res_cnt, amdgpu_bo_handle *resources,
                                   bool secure) {

    ASSERT_NE(resources, nullptr);

    unsigned pm4_dw = packet_size / sizeof(uint32_t);
    ASSERT_TRUE(pm4_dw <= 1024);

    int r;
    amdgpu_cs_ib_info ib_info{};
    amdgpu_cs_request ibs_request{};

    amdgpu_bo_alloc_request req{};
    req.alloc_size = 4096;
    req.phys_alignment = 4096;
    req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;

    BufferObject bo;
    ASSERT_TRUE(dev_.alloc(req, bo) && bo.is_valid());

    // copy PM4 packet to ring from caller
    auto ring_ptr = bo.cpu_address();
    memcpy(ring_ptr, pm4_packet, packet_size);

    ib_info.ib_mc_address = bo.gpu_address();
    ib_info.size = pm4_dw;
    if (secure)
        ib_info.flags |= AMDGPU_IB_FLAGS_SECURE;

    ibs_request.ip_type = ip;
    ibs_request.ring = ring_id;
    ibs_request.number_of_ibs = 1;
    ibs_request.ibs = &ib_info;
    ibs_request.fence_info.handle = nullptr;

    vector<amdgpu_bo_handle> all_res(resources, resources + res_cnt);
    all_res.push_back(bo.handle());

    r = amdgpu_bo_list_create(dev_.handle(), all_res.size(), all_res.data(),
                              nullptr, &ibs_request.resources);
    ASSERT_EQ(r, 0);

    // submit CS
    r = amdgpu_cs_submit(ctx.handle(), 0, &ibs_request, 1);
    ASSERT_EQ(r, 0);

    r = amdgpu_bo_list_destroy(ibs_request.resources);
    ASSERT_EQ(r, 0);

    amdgpu_cs_fence fence_status{};
    fence_status.ip_type = ip;
    fence_status.ip_instance = 0;
    fence_status.ring = ring_id;
    fence_status.context = ctx.handle();
    fence_status.fence = ibs_request.seq_no;

    // wait for IB accomplished
    uint32_t expired = 0;
    r = amdgpu_cs_query_fence_status(&fence_status, AMDGPU_TIMEOUT_INFINITE, 0, &expired);
    ASSERT_EQ(r, 0);
    ASSERT_EQ(expired, 1);
}

TEST_F(BasicTest, GFXCommandSubmission) {
    // write data using the CP
    command_submission_write_linear_helper_with_secure(AMDGPU_HW_IP_GFX, false);
}

TEST_F(BasicTest, GFXDispatchTest) {
    drm_amdgpu_info_hw_ip info{};
    int r;
    uint32_t ring_id;

    r = amdgpu_query_hw_ip_info(dev_.handle(), AMDGPU_HW_IP_GFX, 0, &info);
    ASSERT_EQ(r, 0);

    if (!info.available_rings)
        printf("SKIP ... as there's no graphics ring\n");

    for (ring_id = 0; (1 << ring_id) & info.available_rings; ring_id++) {

    }
}

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
