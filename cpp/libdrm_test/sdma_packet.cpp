#include "sdma_packet.h"
#include "amdgpu_util.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

using namespace std;

static const uint8_t SDMA_SUBOP_COPY_LINEARY = 0;

SDMACopyDataPacket::SDMACopyDataPacket(const void ** const dsts, const void *src, int n, unsigned int size) {
    assert(n <= 2 && "SDMACopyDataPacket does not support more than 2st addresses!");

    vector<const void*> dst(n);
    copy(dsts, dsts + n, begin(dst));

    const int BITS = 21;
    const int TWO_MEG = 1 << BITS;
    const auto single_packet_size = sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_COPY_LINEAR::DST_ADDR[0]) * n;
    bytes = ((size + TWO_MEG - 1) >> BITS) * single_packet_size;
    packet = reinterpret_cast<SDMA_PKT_COPY_LINEAR *>(alloc_data(bytes));
    auto p = packet;

    int32_t s;
    while (size > 0) {
        // SDMA support maximum 0x3fffe0 byte in one copy, take 2M here
        if (size > TWO_MEG)
            s = TWO_MEG;
        else
            s = size;

        memset(p, 0, single_packet_size);
        p->HEADER_UNION.op = static_cast<uint8_t>(SDMA_OP::COPY);
        p->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEARY;
        p->HEADER_UNION.broadcast = n > 1 ? 1 : 0;
        p->COUNT_UNION.count = size - 1;
        split64(reinterpret_cast<uint64_t>(src),
                p->SRC_ADDR_LO_UNION.DW_3_DATA,
                p->SRC_ADDR_HI_UNION.DW_4_DATA);
        for (int i = 0; i < n; i++)
            split64(reinterpret_cast<uint64_t>(dst[i]),
                    p->DST_ADDR[i].DST_ADDR_LO_UNION.DW_5_DATA,
                    p->DST_ADDR[i].DST_ADDR_HI_UNION.DW_6_DATA);

        p = reinterpret_cast<SDMA_PKT_COPY_LINEAR *>(reinterpret_cast<char *>(p) + single_packet_size);

        for (int i = 0; i < n; i++)
            dst[i] = reinterpret_cast<const char *>(dst[i]) + s;
        src = reinterpret_cast<const char *>(src) + s;
        size -= s;
    }
}

SDMACopyDataPacket::SDMACopyDataPacket(const void *dst, const void *src, unsigned int size)
    : SDMACopyDataPacket(&dst, src, 1, size) {
}
