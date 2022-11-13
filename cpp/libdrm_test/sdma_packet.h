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
    SDMACopyDataPacket(const void *dst, const void *src, unsigned size);
    SDMACopyDataPacket(const void ** const dsts, const void *src, int n, unsigned size);
    unsigned size_in_bytes() const override { return bytes; }
    const void *get_packet() const override { return packet; }
private:
    SDMA_PKT_COPY_LINEAR *packet;
    unsigned bytes;
};