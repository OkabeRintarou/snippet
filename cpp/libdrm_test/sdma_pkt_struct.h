#pragma once

#include <cstdint>

struct SDMA_PKT_COPY_LINEAR {
    union {
        struct {
            unsigned op : 8;
            unsigned sub_op : 8;
            unsigned reserved_0 : 11;
            unsigned broadcast : 1;
            unsigned reserved_1 : 4;
        };
        uint32_t DW_O_DATA;
    } HEADER_UNION;

    union {
        struct {
            unsigned count : 22;
            unsigned reserved_0 : 10;
        };
        uint32_t DW_1_DATA;
    } COUNT_UNION;

    union {
        struct {
            unsigned reserved_0 : 16;
            unsigned dst_sw : 2;
            unsigned reserved_1 : 4;
            unsigned dst_ha : 1;
            unsigned reserved_2 : 1;
            unsigned src_sw : 2;
            unsigned reserved_3 : 4;
            unsigned src_ha : 1;
            unsigned reserved_4 : 1;
        };
        uint32_t DW_2_DATA;
    } PARAMETER_UNION;

    union {
        struct {
            unsigned src_addr_31_0 : 32;
        };
        uint32_t DW_3_DATA;
    } SRC_ADDR_LO_UNION;

    union {
        struct {
            unsigned src_addr_63_32 : 32;
        };
        uint32_t DW_4_DATA;
    } SRC_ADDR_HI_UNION;

    struct {
        union {
            struct {
                unsigned dst_addr_31_0 : 32;
            };
            uint32_t DW_5_DATA;
        } DST_ADDR_LO_UNION;

        union {
            struct {
                unsigned dst_addr_63_32 : 32;
            };
            uint32_t DW_6_DATA;
        } DST_ADDR_HI_UNION;
    } DST_ADDR[0];
};
