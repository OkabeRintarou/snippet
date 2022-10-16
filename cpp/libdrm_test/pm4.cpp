#include "pm4.h"
#include <cstring>
#include <iomanip>
#include <iostream>


void BasePacket::dump() const {
    unsigned size = size_in_dwords();
    unsigned i;
    const auto packet = static_cast<const uint32_t *>(get_packet());

    std::cout << "Packet Dump:" << std::hex;
    for (i = 0; i < size; i++)
        std::cout << " " << std::setw(8) << std::setfill('0') << packet[i];
    std::cout << std::endl;
}

void *BasePacket::alloc_data() {
    data.resize(size_in_bytes());
    return data.data();
}

unsigned PM4Packet::calc_count_value() const {
    return size_in_dwords() - sizeof(PM4_TYPE_3_HEADER) / sizeof(uint32_t) - 1;
}

void PM4Packet::init_header(PM4_TYPE_3_HEADER &header, OpCode opcode, int shader_type) {
    header.data = 0;
    header.op_code = static_cast<uint8_t>(opcode);
    header.type = PM4_TYPE_3;
    header.count = calc_count_value();
    header.shader_type = shader_type;
    header.predicate = 0;
}

PM4WriteDataPacket::PM4WriteDataPacket(const PM4WriteDataPacket::Config &conf, uint64_t dst_addr, const uint32_t  *data, unsigned ndw) {
    this->ndw = ndw;
    init_packet(conf, dst_addr, data);
}

unsigned PM4WriteDataPacket::size_in_bytes() const {
    return offsetof(PM4_WRITE_DATA_CI, data) + ndw * sizeof(uint32_t);
}

void PM4WriteDataPacket::init_packet(const PM4WriteDataPacket::Config &conf, uint64_t dst_addr, const uint32_t *data) {
    packet = reinterpret_cast<PM4_WRITE_DATA_CI*>(alloc_data());

    init_header(packet->header, OpCode::WRITE_DATA);
    packet->dst_sel = static_cast<const unsigned>(conf.dst_sel);
    packet->addr_inc = static_cast<const unsigned>(conf.addr_inc);
    packet->write_confirm = static_cast<const unsigned>(conf.write_confirm);
    packet->cache_policy = static_cast<const unsigned>(conf.cache_policy);
    packet->engine_sel = static_cast<const unsigned>(conf.engine_sel);
    packet->dst_addr_lo = packet->dst_addr_hi = 0;

    switch (conf.dst_sel) {
        case MecWriteData::DestSel::MEM_MAPPED_REGISTER:
            packet->dst_addr_lo = 0x3ffff & dst_addr;
            break;
        case MecWriteData::DestSel::MEMORY_SYNC:
        case MecWriteData::DestSel::MEMORY_ASYNC:
        case MecWriteData::DestSel::GDS:
        case MecWriteData::DestSel::TC_L2:
            packet->dst_addr_lo = 0xfffffffc & dst_addr;
            packet->dst_addr_hi = (0xffffffff00000000 & dst_addr) >> 32u;
            break;
    }

    memcpy(packet->data, data, ndw * sizeof(uint32_t));
}
