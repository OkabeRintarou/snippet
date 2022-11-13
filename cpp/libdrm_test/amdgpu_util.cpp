#include "amdgpu_util.h"

void split64(uint64_t src, uint32_t &low32, uint32_t &high32) {
    low32 = static_cast<uint32_t>(src);
    high32 = static_cast<uint32_t>(src >> 32u);
}
