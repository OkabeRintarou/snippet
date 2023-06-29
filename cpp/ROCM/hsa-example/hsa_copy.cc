#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

static hsa_status_t get_agent_callback(hsa_agent_t agent, void *data) {
    hsa_device_type_t device_type;
    assert(hsa_agent_get_info(
            agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);

    if (device_type == HSA_DEVICE_TYPE_GPU) {
        hsa_agent_t *r = (hsa_agent_t *)data;
        *r = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t vram_region_callback(hsa_region_t region, void *data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(
        region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment)
        return HSA_STATUS_SUCCESS;
    
    hsa_region_global_flag_t flags;
    hsa_region_get_info(
        region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
        auto r = static_cast<hsa_region_t*>(data);
        *r = region;
        return HSA_STATUS_INFO_BREAK;
    }
    
    return HSA_STATUS_SUCCESS;
}

static hsa_region_t get_vram_region(hsa_agent_t agent) {
    hsa_region_t vram_region;
    hsa_status_t status =
        hsa_agent_iterate_regions(
            agent, vram_region_callback, &vram_region);
    assert(status == HSA_STATUS_INFO_BREAK);
    return vram_region;
}

static hsa_status_t gtt_region_callback(hsa_region_t region, void *data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(
        region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment)
        return HSA_STATUS_SUCCESS;
    
    hsa_region_global_flag_t flags;
    hsa_region_get_info(
        region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
        auto r = static_cast<hsa_region_t*>(data);
        *r = region;
        return HSA_STATUS_INFO_BREAK;
    }
    
    return HSA_STATUS_SUCCESS;

}

static hsa_region_t get_gtt_region(hsa_agent_t agent) {
    hsa_region_t gtt_region;
    hsa_status_t status =
        hsa_agent_iterate_regions(
            agent, gtt_region_callback, &gtt_region);
    assert(status == HSA_STATUS_INFO_BREAK);
    return gtt_region;
}

static const int BUF_SIZE = 1 << 22;

static void print_info(void *ptr, const char *name) {
    hsa_status_t status;
    hsa_amd_pointer_info_t info = {
        .size = sizeof(hsa_amd_pointer_info_t),
    };
    
    status = hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr);
    assert(status == HSA_STATUS_SUCCESS);
    printf("%s type=%d agentbase=%p hostbase=%p own=%lu\n",
        name, info.type, info.agentBaseAddress, info.hostBaseAddress,
        info.agentOwner.handle);
}

static void copy(void *dst, void *src, const char *name) {
    struct timespec tv1, tv2;
    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    if (hsa_memory_copy(dst, src, BUF_SIZE) != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "%s copy fail!\n", name);
        return;
    }
    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));
    
    double start = tv1.tv_sec * 1e9 + tv1.tv_nsec;
    double end = tv2.tv_sec * 1e9 + tv2.tv_nsec;
    double rate = (double)BUF_SIZE / (end - start);
    printf("%s copy rate %f GB/s\n", name, rate);
}

int main() {
    assert(hsa_init() == HSA_STATUS_SUCCESS); 
    hsa_agent_t gpu_agent;
    hsa_status_t status = hsa_iterate_agents(get_agent_callback, &gpu_agent);
    assert(status == HSA_STATUS_INFO_BREAK);

    hsa_region_t vram_region = get_vram_region(gpu_agent);
    hsa_region_t gtt_region = get_gtt_region(gpu_agent);

    int *gtt_dst;
    status = hsa_memory_allocate(gtt_region, BUF_SIZE, (void**)&gtt_dst);
    assert(status == HSA_STATUS_SUCCESS);
    print_info(gtt_dst, "gtt dst");


    int *gtt_src;
    status = hsa_memory_allocate(gtt_region, BUF_SIZE, (void**)&gtt_src);
    assert(status == HSA_STATUS_SUCCESS);
    print_info(gtt_src, "gtt src");
    for (int i = 0, e = BUF_SIZE / sizeof(int); i < e; i++) {
        gtt_src[i] = i;
    }

    int *vram_dst;
    status = hsa_memory_allocate(vram_region, BUF_SIZE, (void**)&vram_dst);
    assert(status == HSA_STATUS_SUCCESS);
    print_info(vram_dst, "vram dst");

    int *vram_src;
    status = hsa_memory_allocate(vram_region, BUF_SIZE, (void**)&vram_src);
    assert(status == HSA_STATUS_SUCCESS);
    print_info(vram_src, "vram src");

    void *host_dst = nullptr;
    assert(!posix_memalign(&host_dst, 0x200000, BUF_SIZE));
    assert(host_dst != nullptr);
    print_info(host_dst, "host dst");

    void *host_src = nullptr;
    assert(!posix_memalign(&host_src, 0x200000, BUF_SIZE));
    assert(host_src != nullptr);
    print_info(host_src, "host src");

    copy(gtt_dst, vram_src, "P2P");

    for (int i = 0; i < 10; i++) {
        memset(gtt_dst, 0, BUF_SIZE);
        copy(gtt_dst, gtt_src, "P2P");
        if (memcmp(gtt_dst, gtt_src, BUF_SIZE))
            printf("P2P copy content is wrong!\n");
    }

    for (int i = 0; i < 10; i++)
        copy(gtt_dst, vram_src, "D2P");
    for (int i = 0; i < 10; i++)
        copy(vram_dst, gtt_src, "P2D");
    for (int i = 0; i < 10; i++)
        copy(vram_dst, vram_src, "D2D");
    
    for (int i = 0; i < 10; i++)
        copy(host_dst, host_src, "H2H");
    for (int i = 0; i < 10; i++)
        copy(host_dst, gtt_src, "P2H");
    for (int i = 0; i < 10; i++)
        copy(gtt_dst, host_src, "H2P");

    for (int i = 0; i < 10; i++)
        copy(host_dst, vram_src, "D2H");
    for (int i = 0; i < 10; i++)
        copy(vram_dst, host_src, "H2D");

    assert(hsa_shut_down() == HSA_STATUS_SUCCESS);
    return 0;
}
