#pragma once

#include "packet_common.h"
#include "sdma_pkt_struct.h"

enum class SDMA_OP {
    NOP = 0,
    COPY = 1,
    WRITE = 2,
    FENCE = 5,
    TRAP = 6,
    CONST_FILL = 11,
    TIMESTAMP = 13,
};

class SDMAPacket : public BasePacket {
public:
    SDMAPacket() = default;
    ~SDMAPacket() override = default;
    PacketType type() const override { return PacketType::SDMA; }
};

class SDMACopyDataPacket : public SDMAPacket {
public:
    SDMACopyDataPacket(unsigned family_id, void *dst, void *src, unsigned size);
    SDMACopyDataPacket(unsigned family_id, void ** const dsts, void *src, int n, unsigned size);
private:
    SDMA_PKT_COPY_LINEAR *packet;
    unsigned size;
};