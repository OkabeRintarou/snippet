#pragma once

#include <cstdint>
#include <vector>

enum class OpCode {
    NOP                     = 0x10,
    WRITE_DATA              = 0x37,
};

#define PM4_TYPE_0 0
#define PM4_TYPE_2 2
#define PM4_TYPE_3 3

union PM4_TYPE_3_HEADER {
    struct {
        unsigned predicate : 1;     // Predicated version of packet when bit 0 is set
        unsigned shader_type : 1;   // 0: Graphics, 1: Compute Shader
        unsigned reserved : 6;      // Reserved, is set to zero by default
        unsigned op_code : 8;       // IT opcode
        unsigned count : 14;        // Number of DWords - 1 in the information body
        unsigned type : 2;          // Packet identifier, should be 3
    };
    uint32_t data;
};

struct MecWriteData {
    enum class DestSel {
        MEM_MAPPED_REGISTER         = 0,
        MEMORY_SYNC                 = 1,
        TC_L2                       = 2,
        GDS                         = 3,
        MEMORY_ASYNC                = 5,
    };

    enum class AddrInc {
        NO = 0,
        YES = 1,
    };

    enum class WriteConfirm {
        NO = 0,
        YES = 1,
    };

    enum class CachePolicy {
        LRU = 0,
        STREAM = 1,
        BYPASS = 3,
    };

    enum class EngineSel {
        ME = 0,
        PFP = 1,
        CE = 2,
        DE = 3,
    };
};

struct PM4_WRITE_DATA_CI {
    union {
        PM4_TYPE_3_HEADER header;
        uint32_t ordinal1;
    };

    union {
        struct {
            unsigned reserved1 : 8;
            unsigned dst_sel : 4;
            unsigned reserved2 : 4;
            unsigned addr_inc: 1;
            unsigned reserved3 : 3;
            unsigned write_confirm: 1;
            unsigned reserved4 : 4;
            unsigned cache_policy : 2;
            unsigned reverved5 : 3;
            unsigned engine_sel : 2;
        };
        uint32_t ordinal2;
    };

    uint32_t dst_addr_lo;
    uint32_t dst_addr_hi;

    uint32_t data[0];
};

enum class PacketType {
    PM4,
    SDMA,
    AQL,
};

class BasePacket {
public:
    BasePacket() = default;
    virtual ~BasePacket() = default;

    virtual PacketType type() const = 0;
    virtual const void *get_packet() const = 0;
    virtual unsigned size_in_bytes() const = 0;
    unsigned size_in_dwords() const { return size_in_bytes() / sizeof(uint32_t); }

    void dump() const;
protected:
    std::vector<uint8_t> data;

    void *alloc_data();
};

class PM4Packet : public BasePacket {
public:
    PM4Packet() = default;
    virtual ~PM4Packet() = default;

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
    virtual ~PM4WriteDataPacket() = default;
    unsigned size_in_bytes() const override;
    const void *get_packet() const override { return packet; }
private:
    void init_packet(const Config &conf, uint64_t dst_addr, const uint32_t *data);
private:
    unsigned ndw;
    PM4_WRITE_DATA_CI *packet;
};