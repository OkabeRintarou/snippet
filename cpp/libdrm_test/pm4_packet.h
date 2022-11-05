#pragma once

#include "packet_common.h"
#include "pm4_pkt_struct.h"

class PM4Packet : public BasePacket {
public:
    PM4Packet() = default;
    ~PM4Packet() override = default;

    PacketType type() const override { return PacketType::PM4; }
protected:
    unsigned calc_count_value() const;
    void init_header(PM4_TYPE_3_HEADER &header, OpCode opcode, int shader_type = 0);
};

class PM4WriteDataPacket final : public PM4Packet {
public:
    struct Config {
        MecWriteData::DestSel dst_sel;
        MecWriteData::AddrInc addr_inc;
        MecWriteData::WriteConfirm write_confirm;
        MecWriteData::CachePolicy cache_policy;
        MecWriteData::EngineSel engine_sel;
    };
    PM4WriteDataPacket(const Config &conf, uint64_t dst_addr, const uint32_t *data, unsigned ndw);
    ~PM4WriteDataPacket() override = default;
    unsigned size_in_bytes() const override;
    const void *get_packet() const override { return packet; }
private:
    void init_packet(const Config &conf, uint64_t dst_addr, const uint32_t *data);
private:
    unsigned ndw;
    unsigned bytes;
    PM4_WRITE_DATA_CI *packet;
};
