#pragma once
#include <cstdint>
#include <netinet/in.h>

class Network {
public:
    Network() = default;
    ~Network();

    // Create UDP socket, set SO_REUSEADDR, store target address
    bool init(const char* targetIP, uint16_t targetPort);

    // Send one slice (column-major RGB888 RLE):
    //   [byte 0]: slice_id (0-119)
    //   [byte 1]: flags = 0x00 (RLE-RGB888)
    //   [RLE payload]: [count][R][G][B] ...
    // rgba_128x64 points to SLICE_H*SLICE_W*4 bytes, row-major RGBA8
    int sendSlice(uint8_t sliceID, const uint8_t* rgba_128x64);

    // Send one hand frame (row-major RGB888 RLE):
    //   [byte 0]: frame_id (0-255 wrapping)
    //   [byte 1]: flags = 0x01 (2D skeleton)
    //   [RLE payload]: [count][R][G][B] ...
    // bgr_128x64 points to SLICE_H*SLICE_W*3 bytes, row-major BGR8
    int sendHandFrame(uint8_t frameID, const uint8_t* bgr_128x64);

    void shutdown();

private:
    int sendPacket(const uint8_t* buf, int len);

    int               sock_   = -1;
    struct sockaddr_in addr_  = {};
};
