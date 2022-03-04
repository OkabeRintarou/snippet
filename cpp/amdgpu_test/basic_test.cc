#include <alloca.h>
#include <amdgpu.h>
#include <drm.h>
#include <amdgpu_drm.h>
#include <vector>
#include "amdgpu_device.h"
#include "basic_test.h"
#include "result.h"

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

static const uint32_t PACKET_TYPE0 = 0;
static const uint32_t PACKET_TYPE1 = 1;
static const uint32_t PACKET_TYPE2 = 2;
static const uint32_t PACKET_TYPE3 = 3;

static inline constexpr uint32_t CA_PACKET_GET_TYPE(uint32_t h) {
    return (h >> 30) & 3;
}

static inline constexpr uint32_t CA_PACKET_GET_COUNT(uint32_t h) {
    return (h >> 16)  & 0x3fff;
}

static inline constexpr uint32_t CA_PACKET0_GET_REG(uint32_t h) {
    return h & 0xffff;
}

static inline constexpr uint32_t CA_PACKET3_GET_OPCODE(uint32_t h) {
    return (h >> 8) & 0xff;
}

static inline constexpr uint32_t PACKET0(uint32_t reg, uint32_t n) {
    return (PACKET_TYPE0 << 30) | (reg & 0xffff) | ((n & 0x3fff) << 16);
}

static inline constexpr uint32_t PACKET3(uint32_t op, uint32_t n) {
    return (PACKET_TYPE3 << 30) | ((op & 0xff) << 8) | ((n & 0x3fff) << 16);
}

static inline constexpr uint32_t PACKET3_COMPUTE(uint32_t op, uint32_t n) {
    return PACKET3(op, n) | (1 << 1);
}

static const uint32_t PACKET3_NOP = 0x10;
static const uint32_t PACKET3_WRITE_DATA = 0x37;

static inline constexpr uint32_t WRITE_DATA_DST_SEL(uint32_t x) { return x << 8; }
static const uint32_t WR_ONE_ADDR = (1 << 16);
static const uint32_t WR_CONFIRM = (1 << 20);
static inline constexpr uint32_t WRITE_DATA_CACHE_POLIYC(uint32_t x) { return x << 25; }
static inline constexpr uint32_t WRITE_DATA_ENGINE_SEL(uint32_t x) { return x << 30; }

static inline constexpr uint32_t SDMA_PACKET_SI(uint32_t op, uint32_t b, uint32_t t, uint32_t s, uint32_t cnt) {
    return ((op & 0xf) << 28) | 
            ((b & 0x01) << 26) | 
            ((t & 0x01) << 23) |
            ((s & 0x01) << 22) |
            ((cnt & 0xffff) << 0);
}

static inline constexpr uint32_t SDMA_PACKET(uint32_t op, uint32_t sub_op, uint32_t e) {
    return ((e & 0xffff) << 16) | ((sub_op & 0xff) << 8) | ((op & 0xff) << 0);
}

static const uint32_t SDMA_OPCODE_WRITE  = 2;
static const uint32_t SDMA_WRITE_SUB_OPCODE_LINEAR = 0;
static const uint32_t SDMA_WRITE_SUB_OPCODE_TITLED = 1;

static const uint32_t SDMA_OPCODE_COPY = 1;
static const uint32_t SDMA_COPY_SUB_OPCODE_LINEAR = 0;

static const uint32_t SDMA_OPCODE_ATOMIC = 10;

static inline constexpr uint32_t SDMA_ATOMIC_LOOP(uint32_t x) { return x << 0; }
static inline constexpr uint32_t SDMA_ATOMIC_TMZ(uint32_t x) { return x << 2; }
static inline constexpr uint32_t SDMA_ATOMIC_OPCODE(uint32_t x) { return x << 9; }

static const uint32_t GFX_COMPUTE_NOP = 0xffff1000;
static const uint32_t SDMA_NOP = 0x0;

static int fill_pm4_dma(uint32_t *pm4, int size, uint64_t gpu_va, int family_id) {
    int i = 0, j = 0;

    if (family_id == AMDGPU_FAMILY_SI) {
        pm4[i++] = SDMA_PACKET_SI(SDMA_OPCODE_WRITE, 0, 0, 0, size);
    } else {
        pm4[i++] = SDMA_PACKET(SDMA_OPCODE_WRITE, SDMA_WRITE_SUB_OPCODE_LINEAR, 0);
    }
    pm4[i++] = 0xfffffffc & gpu_va;
    pm4[i++] = (0xffffffff00000000 & gpu_va) >> 32;
    if (family_id >= AMDGPU_FAMILY_AI) {
        pm4[i++] = size - 1;
    } else if (family_id != AMDGPU_FAMILY_SI) {
        pm4[i++] = size;
    }
    
    while (j++ < size) {
        pm4[i++] = 0xdeadbeaf;
    }
    return i;
}

static int fill_pm4_gfx_or_compute(uint32_t *pm4, int size, uint64_t gpu_va) {
    int i = 0, j = 0;

    pm4[i++] = PACKET3(PACKET3_WRITE_DATA, 2 + size);
    pm4[i++] = WRITE_DATA_DST_SEL(5) | WR_CONFIRM;
    pm4[i++] = 0xfffffffc & gpu_va;
    pm4[i++] = (0xffffffff00000000 & gpu_va) >> 32;
    while (j++ < size) {
        pm4[i++] = 0xdeadbeaf;
    }

    return i;
}

static int fill_pm4_data(uint32_t *pm4, int size, uint64_t gpu_va, int ip_type, int family_id) {
    if (ip_type == AMDGPU_HW_IP_DMA) {
        return fill_pm4_dma(pm4, size, gpu_va, family_id);
    } else if (ip_type == AMDGPU_HW_IP_GFX || 
               ip_type == AMDGPU_HW_IP_COMPUTE) {
        return fill_pm4_gfx_or_compute(pm4, size, gpu_va);
    }
    return -1;
}

static void amdgpu_test_exec_cs_helper_raw(Device &device, 
                amdgpu_context_handle context_handle,
                unsigned ip_type, int instance, int pm4_dw,
                uint32_t *pm4_src, int res_cnt, 
                amdgpu_bo_handle *resources,
                amdgpu_cs_ib_info *ib_info,
                amdgpu_cs_request *ibs_request) {

    // allocate IB
    Result<BufferObject, int> ib_bo_result = device.alloc_bo(4096);
    EXPECT_TRUE(ib_bo_result.is_ok());
    BufferObject &&bo = ib_bo_result.take_ok_value();
    Result<BufferObject::Ptr, int> ib_ptr_result = bo.mmap();
    EXPECT_TRUE(ib_ptr_result.is_ok());
    BufferObject::Ptr &&ib_result_cpu = ib_ptr_result.take_ok_value();
  
    // copy PM4 packet to ring from caller
    uint32_t *ring_ptr = static_cast<uint32_t*>(ib_result_cpu.ptr());
    memcpy(ring_ptr, pm4_src, pm4_dw * sizeof(*pm4_src));

    ib_info->ib_mc_address = bo.gpu_address();
    ib_info->size = pm4_dw;

    ibs_request->ip_type = ip_type;
    ibs_request->ring = instance;
    ibs_request->number_of_ibs = 1;
    ibs_request->ibs = ib_info;
    ibs_request->fence_info.handle = nullptr;

    amdgpu_bo_handle *all_res = static_cast<amdgpu_bo_handle*>(alloca(sizeof(resources[0]) * (res_cnt + 1)));
    memcpy(all_res, resources, sizeof(resources[0]) * res_cnt);
    all_res[res_cnt] = bo.bo_handle();

    int r;
    r = amdgpu_bo_list_create(device.raw_handle(), res_cnt + 1, all_res,
                                nullptr, &ibs_request->resources);
    EXPECT_EQ(r, 0);

    // submit CS
    r = amdgpu_cs_submit(context_handle, 0, ibs_request, 1);
    EXPECT_EQ(r, 0);

    r = amdgpu_bo_list_destroy(ibs_request->resources);
    EXPECT_EQ(r, 0);
    // wait for IB accomplished
    amdgpu_cs_fence fence_status{0};
    fence_status.ip_type = ip_type;
    fence_status.ip_instance = 0;
    fence_status.ring = ibs_request->ring;
    fence_status.context = context_handle;
    fence_status.fence = ibs_request->seq_no;

    uint32_t expired;
    r = amdgpu_cs_query_fence_status(&fence_status, AMDGPU_TIMEOUT_INFINITE, 0, &expired);
    EXPECT_EQ(r, 0);
    EXPECT_TRUE(expired);
}

TEST_F(BasicTest, CommandSubmissinTest_GFX) {
    drm_amdgpu_info_hw_ip hw_ip_info;
    int r;
    const int ip_type = AMDGPU_HW_IP_GFX;
    const int sdma_write_length = 128;
    const size_t pm4_dw = 256;
    uint64_t gtt_flags[2] = {0, AMDGPU_GEM_CREATE_CPU_GTT_USWC};
    amdgpu_cs_ib_info ib_info;
    amdgpu_cs_request ibs_request;
    std::vector<amdgpu_bo_handle> resources(1);

    memset(&ib_info, 0, sizeof(ib_info));
    memset(&ibs_request, 0, sizeof(ibs_request));

    Device &device = devices_.front();
    r = amdgpu_query_hw_ip_info(device.raw_handle(), ip_type, 0, &hw_ip_info);
    EXPECT_TRUE(!r);
    Result<Context, int> context_result = device.alloc_context();
    EXPECT_TRUE(context_result.is_ok());
    Context&& context = context_result.take_ok_value();
    amdgpu_context_handle context_handle = context.raw_handle();
    EXPECT_TRUE(context_handle != nullptr);
    

    int family_id = device.device_info().gpu_info.family_id;

    for (int ring_id = 0; (1 << ring_id) & hw_ip_info.available_rings; ring_id++) {
        int loop = 0;
        while (loop < 2) {
            Result<BufferObject, int> bo_result = device.alloc_bo(
                        sdma_write_length * sizeof(uint32_t), 4096, 
                        AMDGPU_GEM_DOMAIN_GTT, gtt_flags[loop]);
            EXPECT_TRUE(bo_result.is_ok());

            BufferObject &&bo = bo_result.take_ok_value();
            resources[0] = bo.bo_handle();

            Result<BufferObject::Ptr, int> ptr_result = bo.mmap();
            EXPECT_TRUE(ptr_result.is_ok());

            BufferObject::Ptr&& ptr = ptr_result.take_ok_value();
            EXPECT_TRUE(ptr);

            auto typed_ptr = to_typed_ptr<uint32_t>(ptr);
            typed_ptr.fill(0);

            std::vector<uint32_t> pm4_vec(pm4_dw, 0);
            uint32_t *pm4 = pm4_vec.data();

            // fulfill pm4: test DMA write-linear
            int data_size = fill_pm4_data(pm4, sdma_write_length, bo.gpu_address(), ip_type, family_id);
            EXPECT_TRUE(data_size > 0);

            amdgpu_test_exec_cs_helper_raw(
                    device, context_handle,
                    ip_type, ring_id, data_size, pm4,
                    1, resources.data(), &ib_info, &ibs_request);
            // verify if SDMA test result meets with expected
            for (int i = 0; i < sdma_write_length; i++) {
                EXPECT_EQ(typed_ptr[i], 0xdeadbeaf);
            }
            ++loop;
        }
    }
}

