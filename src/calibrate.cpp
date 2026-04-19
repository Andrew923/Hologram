#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <getopt.h>

#include "engine/Renderer.h"
#include "engine/Slicer.h"
#include "engine/Network.h"

static volatile sig_atomic_t g_running = 1;

static void sigHandler(int /*sig*/) { g_running = 0; }

static void usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --ip <addr>            Target IP (default: 10.42.0.169)\n"
            "  --port <port>          Target UDP port (default: 4210)\n"
            "  --panel-offset <px>    Panel offset in voxels (default: 12)\n"
            "  --offset-sign <v>      Offset sign (+1 or -1, default: 1)\n"
            "  --sweep-dir <v>        Sweep direction (+1 or -1, default: 1)\n"
            "  --swap-sin-cos         Swap sin/cos placement\n"
            "  --phase-offset <rad>   Phase offset in radians (default: 0)\n"
            "  --sleep-us <us>        Delay between slice sends (default: 100)\n"
            "  --help                 Show this message\n",
            prog);
}

static void buildCalibrationCube(uint8_t* voxels, size_t bytes)
{
    memset(voxels, 0, bytes);

    static constexpr int kCubeSide = 8;
    static constexpr int kMargin = 1;
    const int startX = VOXEL_W - kCubeSide - kMargin;
    const int startY = VOXEL_H - kCubeSide - kMargin;
    const int startZ = VOXEL_D - kCubeSide - kMargin;

    for (int z = startZ; z < startZ + kCubeSide; ++z) {
        for (int y = startY; y < startY + kCubeSide; ++y) {
            for (int x = startX; x < startX + kCubeSide; ++x) {
                const size_t idx = ((size_t)z * VOXEL_H + (size_t)y) * VOXEL_W + (size_t)x;
                voxels[idx * 4 + 0] = 255;
                voxels[idx * 4 + 1] = 255;
                voxels[idx * 4 + 2] = 255;
                voxels[idx * 4 + 3] = 255;
            }
        }
    }

    fprintf(stderr,
            "calibrate: cube side=%d at x=[%d..%d] y=[%d..%d] z=[%d..%d]\n",
            kCubeSide,
            startX, startX + kCubeSide - 1,
            startY, startY + kCubeSide - 1,
            startZ, startZ + kCubeSide - 1);
}

int main(int argc, char* argv[])
{
    char targetIP[64] = "10.42.0.169";
    uint16_t targetPort = 4210;
    float panelOffset = (float)PANEL_OFFSET;
    float offsetSign = 1.0f;
    float sweepDir = 1.0f;
    bool swapSinCos = false;
    float phaseOffset = 0.0f;
    int sleepUs = 100;

    static const struct option longopts[] = {
        {"ip", required_argument, nullptr, 'i'},
        {"port", required_argument, nullptr, 'p'},
        {"panel-offset", required_argument, nullptr, 'o'},
        {"offset-sign", required_argument, nullptr, 's'},
        {"sweep-dir", required_argument, nullptr, 'd'},
        {"swap-sin-cos", no_argument, nullptr, 'w'},
        {"phase-offset", required_argument, nullptr, 'f'},
        {"sleep-us", required_argument, nullptr, 'u'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "i:p:o:s:d:wf:u:h", longopts, nullptr)) != -1) {
        switch (c) {
            case 'i': snprintf(targetIP, sizeof(targetIP), "%s", optarg); break;
            case 'p': targetPort = (uint16_t)atoi(optarg); break;
            case 'o': panelOffset = strtof(optarg, nullptr); break;
            case 's': offsetSign = strtof(optarg, nullptr); break;
            case 'd': sweepDir = strtof(optarg, nullptr); break;
            case 'w': swapSinCos = true; break;
            case 'f': phaseOffset = strtof(optarg, nullptr); break;
            case 'u': sleepUs = atoi(optarg); break;
            case 'h': usage(argv[0]); return 0;
            default: usage(argv[0]); return 1;
        }
    }

    if (offsetSign == 0.0f) {
        fprintf(stderr, "calibrate: --offset-sign 0 is invalid, using 1\n");
        offsetSign = 1.0f;
    }
    if (sweepDir == 0.0f) {
        fprintf(stderr, "calibrate: --sweep-dir 0 is invalid, using 1\n");
        sweepDir = 1.0f;
    }
    if (sleepUs < 0) {
        fprintf(stderr, "calibrate: --sleep-us cannot be negative, using 0\n");
        sleepUs = 0;
    }

    fprintf(stderr,
            "calibrate: ip=%s port=%u panelOffset=%.3f offsetSign=%.3f "
            "sweepDir=%.3f swapSinCos=%d phaseOffset=%.3f sleepUs=%d\n",
            targetIP, targetPort, panelOffset, offsetSign, sweepDir,
            swapSinCos ? 1 : 0, phaseOffset, sleepUs);

    signal(SIGTERM, sigHandler);
    signal(SIGINT, sigHandler);

    Renderer renderer;
    if (!renderer.init()) {
        fprintf(stderr, "calibrate: Renderer::init failed\n");
        return 1;
    }

    Slicer slicer;
    if (!slicer.init("shaders/slice_compute_calibrate.glsl")) {
        fprintf(stderr, "calibrate: Slicer::init failed\n");
        return 1;
    }
    slicer.setCalibrationParams(panelOffset, offsetSign, sweepDir, swapSinCos, phaseOffset);

    Network network;
    if (!network.init(targetIP, targetPort)) {
        fprintf(stderr, "calibrate: Network::init failed\n");
        return 1;
    }

    static uint8_t voxels[VOXEL_BYTES];
    buildCalibrationCube(voxels, sizeof(voxels));
    renderer.uploadVoxelBuffer(voxels);

    SliceBuffer sliceBuffer = {};

    slicer.kickDispatch(renderer.getVoxelTextureID());
    while (g_running) {
        slicer.syncReadback(sliceBuffer);

        for (int i = 0; i < SLICE_COUNT; ++i) {
            network.sendSlice((uint8_t)i, &sliceBuffer.data[i][0][0][0]);
            if (sleepUs > 0) usleep((useconds_t)sleepUs);
        }

        slicer.kickDispatch(renderer.getVoxelTextureID());
    }

    slicer.syncReadback(sliceBuffer);
    network.shutdown();
    slicer.shutdown();
    renderer.shutdown();
    return 0;
}
