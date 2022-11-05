#pragma once

#include <cstdint>
#include <vector>

enum class OpCode {
    NOP                     = 0x10,
    WRITE_DATA              = 0x37,
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

    void *alloc_data(size_t bytes);
};