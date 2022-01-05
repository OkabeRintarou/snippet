#include <amdgpu.h>
#include <drm.h>
#include <amdgpu_drm.h>
#include "amdgpu_device.h"
#include "bo_test.h"

using namespace amdgpu;

bool BoTest::init() {
	return Devices::load(devices_);
}

bool BoTest::fini() {
    return Devices::unload(devices_);
}

TEST_F(BoTest, MapUnmapTest) {

    EXPECT_FALSE(devices_.empty());

    Device &device = devices_.front();
    
    const uint64_t BUFFER_SIZE = 4096;

    Result<BufferObject, int> bor = device.alloc_bo(BUFFER_SIZE);
    EXPECT_TRUE(bor.is_ok());

    BufferObject&& bo = bor.take_ok_value();
    Result<BufferObject::Ptr, int> ptr_result = bo.mmap();
    EXPECT_TRUE(ptr_result.is_ok());
        
    BufferObject::Ptr&& ptr = ptr_result.take_ok_value();
    EXPECT_TRUE(ptr);

    auto typed_ptr = to_typed_ptr<uint32_t>(ptr);
    EXPECT_TRUE(typed_ptr);

    for (int i = 0; i < typed_ptr.size(); i++) {
        typed_ptr[i] = 0xdeadbeaf;
    }
}
