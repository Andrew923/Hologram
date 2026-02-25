#include "Network.h"
#include "Slicer.h"   // for SLICE_W, SLICE_H
#include <cstdio>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

Network::~Network() { shutdown(); }

bool Network::init(const char* targetIP, uint16_t targetPort)
{
    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0) {
        perror("Network: socket");
        return false;
    }

    int opt = 1;
    setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr_, 0, sizeof(addr_));
    addr_.sin_family      = AF_INET;
    addr_.sin_port        = htons(targetPort);
    if (inet_pton(AF_INET, targetIP, &addr_.sin_addr) <= 0) {
        fprintf(stderr, "Network: invalid IP: %s\n", targetIP);
        close(sock_);
        sock_ = -1;
        return false;
    }

    fprintf(stderr, "Network: will send to %s:%u\n", targetIP, targetPort);
    return true;
}

int Network::sendPacket(const uint8_t* buf, int len)
{
    return (int)sendto(sock_, buf, len, 0,
                       (struct sockaddr*)&addr_, sizeof(addr_));
}

// -----------------------------------------------------------------------
// RGB888 RLE helper: column-major scan of an RGBA 128x64 buffer
// RLE format: [count(1)] [R(1)] [G(1)] [B(1)]  — count in [1,255]
// -----------------------------------------------------------------------
static int rleEncodeColMajor(const uint8_t* rgba, int w, int h,
                              uint8_t* out, int outCap)
{
    int n = 0;
    uint8_t cr = 0, cg = 0, cb = 0;
    uint8_t cnt = 0;

    auto flush = [&]() -> bool {
        if (n + 4 > outCap) return false;
        out[n++] = cnt;
        out[n++] = cr;
        out[n++] = cg;
        out[n++] = cb;
        cnt = 0;
        return true;
    };

    bool first = true;
    for (int col = 0; col < w; ++col) {
        for (int row = 0; row < h; ++row) {
            const uint8_t* px = rgba + (row * w + col) * 4;
            uint8_t r = px[0], g = px[1], b = px[2];
            if (first) {
                cr = r; cg = g; cb = b; cnt = 1; first = false;
            } else if (r == cr && g == cg && b == cb && cnt < 255) {
                cnt++;
            } else {
                if (!flush()) return -1;
                cr = r; cg = g; cb = b; cnt = 1;
            }
        }
    }
    if (cnt > 0 && !flush()) return -1;
    return n;
}

// -----------------------------------------------------------------------
// RGB888 RLE helper: row-major scan of a BGR 128x64 buffer
// -----------------------------------------------------------------------
static int rleEncodeRowMajorBGR(const uint8_t* bgr, int w, int h,
                                 uint8_t* out, int outCap)
{
    int n = 0;
    uint8_t cr = 0, cg = 0, cb = 0;
    uint8_t cnt = 0;

    auto flush = [&]() -> bool {
        if (n + 4 > outCap) return false;
        out[n++] = cnt;
        out[n++] = cr;
        out[n++] = cg;
        out[n++] = cb;
        cnt = 0;
        return true;
    };

    bool first = true;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            // BGR storage: index 0=B,1=G,2=R
            const uint8_t* px = bgr + (row * w + col) * 3;
            uint8_t r = px[2], g = px[1], b = px[0];  // convert BGR->RGB
            if (first) {
                cr = r; cg = g; cb = b; cnt = 1; first = false;
            } else if (r == cr && g == cg && b == cb && cnt < 255) {
                cnt++;
            } else {
                if (!flush()) return -1;
                cr = r; cg = g; cb = b; cnt = 1;
            }
        }
    }
    if (cnt > 0 && !flush()) return -1;
    return n;
}

int Network::sendSlice(uint8_t sliceID, const uint8_t* rgba_128x64)
{
    // Max uncompressed: 128*64*3 + 2 header + RLE overhead (4 bytes/run, worst case)
    // Generous buffer: 128*64*4 + 2
    static uint8_t pkt[2 + SLICE_W * SLICE_H * 4 + 8];

    pkt[0] = sliceID;
    pkt[1] = 0x00;  // flags: RLE-RGB888

    int rleLen = rleEncodeColMajor(rgba_128x64, SLICE_W, SLICE_H,
                                   pkt + 2, (int)sizeof(pkt) - 2);
    if (rleLen < 0) {
        fprintf(stderr, "Network: sendSlice RLE buffer overflow\n");
        return -1;
    }

    return sendPacket(pkt, 2 + rleLen);
}

int Network::sendHandFrame(uint8_t frameID, const uint8_t* bgr_128x64)
{
    static uint8_t pkt[2 + SLICE_W * SLICE_H * 4 + 8];

    pkt[0] = frameID;
    pkt[1] = 0x01;  // flags: 2D skeleton

    int rleLen = rleEncodeRowMajorBGR(bgr_128x64, SLICE_W, SLICE_H,
                                      pkt + 2, (int)sizeof(pkt) - 2);
    if (rleLen < 0) {
        fprintf(stderr, "Network: sendHandFrame RLE buffer overflow\n");
        return -1;
    }

    return sendPacket(pkt, 2 + rleLen);
}

void Network::shutdown()
{
    if (sock_ >= 0) { close(sock_); sock_ = -1; }
}
