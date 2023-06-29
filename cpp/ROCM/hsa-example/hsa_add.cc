#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

using namespace std;

static hsa_status_t get_agent_callback(hsa_agent_t agent, void *data) {
    hsa_device_type_t device_type;
    assert(hsa_agent_get_info(
            agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);

    if (device_type == HSA_DEVICE_TYPE_GPU) {
        char agent_name[64];
        assert(hsa_agent_get_info(
            agent, HSA_AGENT_INFO_NAME, agent_name) == HSA_STATUS_SUCCESS);
        char vendor_name[64];
        assert(hsa_agent_get_info(
            agent, HSA_AGENT_INFO_VENDOR_NAME, vendor_name) == HSA_STATUS_SUCCESS);

        printf("%s - %s\n", vendor_name, agent_name);

        hsa_agent_feature_t feature;
        assert(hsa_agent_get_info(
            agent, HSA_AGENT_INFO_FEATURE, &feature) == HSA_STATUS_SUCCESS);
        if (!(feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
            printf("agent not support kernel dispatch!\n");
            return HSA_STATUS_SUCCESS;
        }

#define GET_AGENT_INFO(var, info)   \
        uint32_t var = 0;           \
        assert(hsa_agent_get_info(  \
            agent, info, &var) == HSA_STATUS_SUCCESS);

        GET_AGENT_INFO(wavefront_size,      HSA_AGENT_INFO_WAVEFRONT_SIZE);
        GET_AGENT_INFO(workgroup_max_dim,   HSA_AGENT_INFO_WORKGROUP_MAX_DIM);
        GET_AGENT_INFO(workgroup_max_size,  HSA_AGENT_INFO_WORKGROUP_MAX_SIZE);
        GET_AGENT_INFO(grid_max_dim,        HSA_AGENT_INFO_GRID_MAX_DIM);
        GET_AGENT_INFO(grid_max_size,       HSA_AGENT_INFO_GRID_MAX_SIZE);
        GET_AGENT_INFO(cache_size,          HSA_AGENT_INFO_CACHE_SIZE);
        GET_AGENT_INFO(queue_max,           HSA_AGENT_INFO_QUEUES_MAX);
        GET_AGENT_INFO(queue_min_size,      HSA_AGENT_INFO_QUEUE_MIN_SIZE);
        GET_AGENT_INFO(queue_max_size,      HSA_AGENT_INFO_QUEUE_MAX_SIZE);

        printf("\t wavefront size: %u\n", wavefront_size);
        printf("\t workgroup max dim: %u\n", workgroup_max_dim);
        printf("\t workgroup max size: %u\n", workgroup_max_size);
        printf("\t grid max dim: %u\n", grid_max_dim);
        printf("\t grid max size: %u\n", grid_max_size);
        printf("\t cache size: %u KB\n", cache_size / 1024);
        printf("\t queue max: %u\n", queue_max);
        printf("\t queue min size: %u\n", queue_min_size);
        printf("\t queue max size: %u\n", queue_max_size);

#undef GET_AGENT_INFO

        hsa_agent_t *r = (hsa_agent_t *)data;
        *r = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

struct symbol_data {
	hsa_executable_symbol_t symbol;
	string name;
};

static hsa_status_t iterate_symbols_callback(
		hsa_executable_t exe,
		hsa_executable_symbol_t symbol,
		void *data) {

	uint32_t symbol_length = 0;
	assert(hsa_executable_symbol_get_info(
				symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
				&symbol_length) == HSA_STATUS_SUCCESS);

	symbol_data *sd = (symbol_data*)data;
	sd->name.resize(symbol_length + 1);
	assert(hsa_executable_symbol_get_info(
				symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME,
				sd->name.data()) == HSA_STATUS_SUCCESS);
	sd->symbol = symbol;

	return HSA_STATUS_INFO_BREAK;
}

static void load_device_kernel(
		hsa_agent_t agent, hsa_kernel_dispatch_packet_t &packet) {
	char name[64];
    assert(hsa_agent_get_info(
            	agent, HSA_AGENT_INFO_NAME, name) == HSA_STATUS_SUCCESS);

	char filename[128];
	snprintf(filename, 128, "vec_add.%s.co", name);
	int fd = open(filename, O_RDWR);
	assert(fd >= 0);	

	hsa_code_object_reader_t reader;
	assert(hsa_code_object_reader_create_from_file(fd, &reader) == HSA_STATUS_SUCCESS);

	hsa_executable_t exe;
	assert(hsa_executable_create_alt(
				HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
				nullptr, &exe) == HSA_STATUS_SUCCESS);

	assert(hsa_executable_load_agent_code_object(
				exe, agent, reader, nullptr, nullptr) == HSA_STATUS_SUCCESS);
	assert(hsa_executable_freeze(exe, nullptr) == HSA_STATUS_SUCCESS);
	uint32_t valid = 0xff;
	assert(hsa_executable_validate(exe, &valid) == HSA_STATUS_SUCCESS);
	assert(valid == 0);

	symbol_data sd;
	assert(hsa_executable_iterate_symbols(
				exe, iterate_symbols_callback, 
				&sd) == HSA_STATUS_INFO_BREAK);
	assert(!sd.name.empty());
	printf("Symbol name: %s\n", sd.name.c_str());

	assert(hsa_executable_symbol_get_info(
				sd.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
				&packet.kernel_object) == HSA_STATUS_SUCCESS);
	assert(hsa_executable_symbol_get_info(
				sd.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
				&packet.group_segment_size) == HSA_STATUS_SUCCESS);

	printf("Group segment size: %u\n", packet.group_segment_size);

	assert(hsa_code_object_reader_destroy(reader) == HSA_STATUS_SUCCESS);
}

struct memory_regions {
	hsa_region_t gtt;
	hsa_region_t vis_vram;
	hsa_region_t invis_vram;
	hsa_region_t kernarg;

	memory_regions() {
		memset(&gtt, 0, sizeof(gtt));
		memset(&vis_vram, 0, sizeof(vis_vram));
		memset(&invis_vram, 0, sizeof(invis_vram));
		memset(&kernarg, 0, sizeof(kernarg));
	}
};

hsa_status_t get_kernarg(hsa_region_t region, void *data) {
	hsa_region_segment_t segment;
	hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
	if (segment != HSA_REGION_SEGMENT_GLOBAL)
		return HSA_STATUS_SUCCESS;

	hsa_region_global_flag_t flags;
	assert(hsa_region_get_info(
				region, HSA_REGION_INFO_GLOBAL_FLAGS, 
				&flags) == HSA_STATUS_SUCCESS);
	
	bool host_accessible = false;
	assert(hsa_region_get_info(
				region, (hsa_region_info_t)HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
				&host_accessible) == HSA_STATUS_SUCCESS);

	auto mr = (memory_regions*)data;

#define GET_REGION_SIZE()			\
	size_t region_size = 0;   		\
	assert(hsa_region_get_info(		\
			region, HSA_REGION_INFO_SIZE, 	\
			&region_size) ==				\
			HSA_STATUS_SUCCESS)

	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
		GET_REGION_SIZE();
		printf("Region kern arg size: 0x%lx\n", region_size);
		mr->kernarg = region;
	} else if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
		GET_REGION_SIZE();
		if (host_accessible) {
			printf("Region visible vram size: 0x%lx\n", region_size);
			mr->vis_vram = region;
		} else {
			printf("Region invisible vram size: 0x%lx\n", region_size);
			mr->invis_vram = region;
		}
	} else if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
		GET_REGION_SIZE();
		printf("Region gtt size: 0x%lx\n", region_size);
		mr->gtt = region;
	}
#undef GET_REGION_SIZE

	return HSA_STATUS_SUCCESS;
}

void init_memory_regions(hsa_agent_t agent, memory_regions &regions) {
	hsa_agent_iterate_regions(agent, get_kernarg, &regions);
}

static void queue_error_callback(hsa_status_t status, hsa_queue_t *queue, void *data) {
	const char *err;

	hsa_status_string(status, &err);
	printf("Error at queue %lu: %s\n", queue->id, err);
}

static const int WIDTH = 1024;
static const int HEIGHT = 1024;
static const int NUM = WIDTH * HEIGHT;

static const int THREAD_PER_BLOCK_X = 16;
static const int THREAD_PER_BLOCK_Y = 16;
static const int THREAD_PER_BLOCK_Z = 1;

float *host_A, *host_B, *host_C;
float *device_A, *device_B, *device_C;

static void init_input(const memory_regions &regions) {
	host_A = new float[NUM];
	host_B = new float[NUM];
	host_C = new float[NUM];

	for (int i = 0; i < NUM; i++) {
		host_B[i] = (float)i;
		host_C[i] = (float)i * 100.0f;
	}
	memset(host_A, 0, NUM * sizeof(float));

	hsa_region_t vram = regions.invis_vram;

	assert(hsa_memory_allocate(vram, NUM * sizeof(float), (void**)&device_A) == HSA_STATUS_SUCCESS);
	assert(hsa_memory_allocate(vram, NUM * sizeof(float), (void**)&device_B) == HSA_STATUS_SUCCESS);
	assert(hsa_memory_allocate(vram, NUM * sizeof(float), (void**)&device_C) == HSA_STATUS_SUCCESS);

	assert(hsa_memory_copy(device_A, host_A, NUM * sizeof(float)) == HSA_STATUS_SUCCESS);
	assert(hsa_memory_copy(device_B, host_B, NUM * sizeof(float)) == HSA_STATUS_SUCCESS);
	assert(hsa_memory_copy(device_C, host_C, NUM * sizeof(float)) == HSA_STATUS_SUCCESS);
}

static void init_packet(hsa_agent_t &agent, const memory_regions &regions, hsa_kernel_dispatch_packet_t *packet) {

	memset((uint8_t*)packet + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);

	packet->workgroup_size_x = THREAD_PER_BLOCK_X;
	packet->workgroup_size_y = THREAD_PER_BLOCK_Y;
	packet->workgroup_size_z = THREAD_PER_BLOCK_Z;

	packet->grid_size_x = WIDTH;
	packet->grid_size_y = HEIGHT;
	packet->grid_size_z = 1;

	printf("Grid: (%u, %u, %u), Block: (%u, %u, %u)\n",
			packet->grid_size_x / packet->workgroup_size_x,
			packet->grid_size_y / packet->workgroup_size_y, 
			packet->grid_size_z,
			packet->workgroup_size_x,
			packet->workgroup_size_y,
			packet->workgroup_size_z);

	struct kern_arg {
		float *a;
		float *b;
		float *c;
		uint32_t width, height;
		uint32_t hidden_block_count_x;
		uint32_t hidden_block_count_y;
		uint32_t hidden_block_count_z;
		uint16_t hidden_group_size_x;
		uint16_t hidden_group_size_y;
		uint16_t hidden_group_size_z;
		uint16_t hidden_remainder_x;
		uint16_t hidden_remainder_y;
		uint16_t hidden_remainder_z;
		uint64_t hidden_global_offset_x;
		uint64_t hidden_global_offset_y;
		uint64_t hidden_global_offset_z;
		uint16_t hidden_grid_dims;
	} *args = nullptr;

	assert(hsa_memory_allocate(regions.kernarg, sizeof(*args), (void**)&args) == HSA_STATUS_SUCCESS);
	packet->kernarg_address = args;

	init_input(regions);

	memset(args, 0, sizeof(*args));

	args->a = device_A;
	args->b = device_B;
	args->c = device_C;

	args->width = WIDTH;
	args->height = HEIGHT;

	args->hidden_block_count_x = WIDTH / THREAD_PER_BLOCK_X;
	args->hidden_block_count_y = WIDTH / THREAD_PER_BLOCK_Y;
	args->hidden_block_count_z = WIDTH / THREAD_PER_BLOCK_Z;
	args->hidden_group_size_x = THREAD_PER_BLOCK_X;
	args->hidden_group_size_y = THREAD_PER_BLOCK_Y;
	args->hidden_group_size_z = THREAD_PER_BLOCK_Z;

	load_device_kernel(agent, *packet);

	uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

	uint16_t setup = 2 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

	__atomic_store_n((uint32_t*)packet, header | (setup << 16), __ATOMIC_RELEASE);
}

int main() {
    assert(hsa_init() == HSA_STATUS_SUCCESS);

    hsa_status_t status;

    hsa_agent_t gpu_agent;
    status = hsa_iterate_agents(get_agent_callback, &gpu_agent);
    assert(status == HSA_STATUS_INFO_BREAK);

	memory_regions regions;
	init_memory_regions(gpu_agent, regions);

    // create user mode queue
    hsa_queue_t *queue = nullptr;
    status = hsa_queue_create(
                gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                queue_error_callback, nullptr,
                UINT32_MAX, UINT32_MAX, &queue);
    assert(status == HSA_STATUS_SUCCESS);

	// Atomically request a new packet ID
	uint64_t packet_id = hsa_queue_load_write_index_relaxed(queue);

	// Wait until the queue is not full before writing the packet
	while (packet_id + 1 - hsa_queue_load_read_index_scacquire(queue) >= queue->size)
		;

	auto p = (hsa_kernel_dispatch_packet_t*)queue->base_address + packet_id;

	init_packet(gpu_agent, regions, p);

	assert(hsa_signal_create(1, 0, nullptr, &p->completion_signal) == HSA_STATUS_SUCCESS);

	hsa_queue_store_write_index_relaxed(queue, packet_id + 1);

	hsa_signal_store_screlease(queue->doorbell_signal, packet_id);

	while (hsa_signal_wait_scacquire(p->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0)
		;
	
	// verify the results
	assert(hsa_memory_copy(host_A, device_A, NUM * sizeof(float)) == HSA_STATUS_SUCCESS);

	int errors = 0;
	for (int i = 0; i < NUM; i++)
		if (host_A[i] != host_B[i] + host_C[i])
			++errors;

	if (errors != 0)
		printf("FAILED: %d errors!\n", errors);
	else
		printf("PASSED!\n");

	delete[] host_A;
	delete[] host_B;
	delete[] host_C;
	hsa_memory_free(device_A);
	hsa_memory_free(device_B);
	hsa_memory_free(device_C);

	hsa_signal_destroy(p->completion_signal);
	hsa_queue_destroy(queue);

    assert(hsa_shut_down() == HSA_STATUS_SUCCESS);
    return 0;
}
