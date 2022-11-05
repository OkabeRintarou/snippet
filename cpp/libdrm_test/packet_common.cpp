#include "packet_common.h"
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

void *BasePacket::alloc_data(size_t bytes) {
    data.resize(bytes);
    return data.data();
}


