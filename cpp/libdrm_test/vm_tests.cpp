#include "amdgpu_test.h"
#include "vm_tests.h"

using namespace amdgpu;

void VMTest::SetUp() {
    EXPECT_TRUE(dev_.is_valid());
}

TEST_F(VMTest, ReserveVMIDTest) {
    struct amdgpu_gpu_info gpu_info {};
    int r = amdgpu_query_gpu_info(dev_.handle(), &gpu_info);
    EXPECT_EQ(r, 0);

    uint32_t gc_ip_type = asic_is_gfx_pipe_removed(gpu_info.family_id,
                                                   gpu_info.chip_external_rev,
                                                   gpu_info.chip_rev) ?
                              AMDGPU_HW_IP_COMPUTE : AMDGPU_HW_IP_GFX;

    amdgpu_bo_alloc_request req {};
    req.alloc_size = 4096;
    req.phys_alignment = 4096;
    req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;

    BufferObject bo;
    Context ctx;

    EXPECT_TRUE(dev_.alloc(req, bo, true) && bo.is_valid());
    EXPECT_TRUE(dev_.alloc(ctx) && ctx.is_valid());

    uint32_t flags = 0;
    r = amdgpu_vm_reserve_vmid(dev_.handle(), flags);
    EXPECT_EQ(r, 0);

    amdgpu_bo_list_handle bo_list;
    r = amdgpu_get_bo_list(dev_.handle(), bo.handle(), nullptr, &bo_list);
    EXPECT_EQ(r, 0);

    auto ptr = static_cast<uint32_t*>(bo.cpu_address());
    for (int i = 0; i < 16; i++) {
        ptr[i] = 0xffff1000;
    }

    amdgpu_cs_ib_info ib_info{};
    amdgpu_cs_request ibs_request{};

    ib_info.ib_mc_address = bo.gpu_address();
    ib_info.size = 16;

    ibs_request.ip_type = gc_ip_type;
    ibs_request.ring = 0;
    ibs_request.number_of_ibs = 1;
    ibs_request.ibs = &ib_info;
    ibs_request.resources = bo_list;
    ibs_request.fence_info.handle = nullptr;

    r = amdgpu_cs_submit(ctx.handle(), 0, &ibs_request, 1);
    EXPECT_EQ(r, 0);

    amdgpu_cs_fence fence{};
    fence.context = ctx.handle();
    fence.ip_type = gc_ip_type;
    fence.ip_instance = 0;
    fence.ring = 0;
    fence.fence = ibs_request.seq_no;

    uint32_t expired;
    r = amdgpu_cs_query_fence_status(&fence, AMDGPU_TIMEOUT_INFINITE, 0, &expired);
    EXPECT_EQ(r, 0);

    r = amdgpu_bo_list_destroy(bo_list);
    EXPECT_EQ(r, 0);

    flags = 0;
    r = amdgpu_vm_unreserve_vmid(dev_.handle(), flags);
    EXPECT_EQ(r, 0);
}

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}