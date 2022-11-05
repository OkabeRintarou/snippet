#pragma once

#include <cstdint>

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