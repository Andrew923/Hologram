#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <getopt.h>

#include "../shared_defs.h"
#include "engine/Renderer.h"
#include "engine/Slicer.h"
#include "engine/Network.h"
#include "engine/InputBridge.h"
#include "application/IApplication.h"
#include "application/CubeApp.h"
#include "application/HandApp.h"
#include "application/PongApp.h"
#include "application/WireframeApp.h"
#include "application/FluidApp.h"

// -----------------------------------------------------------------------
// Globals for signal handling
// -----------------------------------------------------------------------
static volatile sig_atomic_t g_running   = 1;
static pid_t                 g_dockerPid = -1;

static void sigHandler(int /*sig*/)
{
    g_running = 0;
    if (g_dockerPid > 0) {
        kill(g_dockerPid, SIGTERM);
    }
}

// -----------------------------------------------------------------------
// Docker sidecar launch
// -----------------------------------------------------------------------
static pid_t launchDocker()
{
    const char* args[] = {
        "docker", "run", "--runtime", "nvidia", "--rm", "--network", "host",
        "--ipc=host",          // REQUIRED: shares /dev/shm with host
        "--privileged",
        "-v", "/var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket",
        "-v", "/root/jetson-containers/data:/data",
        "-v", "/root/Hologram:/Hologram",
        "--device", "/dev/video0",
        "hand-pose-v5",
        "python3", "/Hologram/python/hand_tracker.py",
        nullptr
    };

    pid_t pid = fork();
    if (pid == 0) {
        // Child: exec docker
        execvp("docker", (char* const*)args);
        perror("execvp docker");
        _exit(1);
    }
    if (pid < 0) {
        perror("fork");
        return -1;
    }
    fprintf(stderr, "main: launched docker sidecar (pid %d)\n", pid);
    return pid;
}

// -----------------------------------------------------------------------
// CLI usage
// -----------------------------------------------------------------------
static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s --app <cube|hand|pong|wireframe|fluid> --ip <pi_ip> --port <port>\n"
        "  --app       Application to run (default: cube)\n"
        "  --ip        Target IP address of Raspberry Pi (default: 192.168.1.100)\n"
        "  --port      Target UDP port (default: 4210)\n"
        "  --no-docker Skip launching the Docker hand tracker sidecar\n",
        prog);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Defaults
    char     appName[32]   = "cube";
    char     targetIP[64]  = "192.168.1.100";
    uint16_t targetPort    = 4210;
    bool     noDocker      = false;

    // Parse CLI
    static const struct option longopts[] = {
        {"app",       required_argument, nullptr, 'a'},
        {"ip",        required_argument, nullptr, 'i'},
        {"port",      required_argument, nullptr, 'p'},
        {"no-docker", no_argument,       nullptr, 'n'},
        {"help",      no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "a:i:p:nh", longopts, nullptr)) != -1) {
        switch (c) {
            case 'a': strncpy(appName, optarg, sizeof(appName)-1); break;
            case 'i': strncpy(targetIP, optarg, sizeof(targetIP)-1); break;
            case 'p': targetPort = (uint16_t)atoi(optarg); break;
            case 'n': noDocker = true; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    fprintf(stderr, "main: app=%s ip=%s port=%u\n", appName, targetIP, targetPort);

    // Install signal handlers
    signal(SIGTERM, sigHandler);
    signal(SIGINT,  sigHandler);

    // 1. C++ creates and zeros the shared memory BEFORE spawning Python
    InputBridge inputBridge;
    if (!inputBridge.init(/*creator=*/true)) {
        fprintf(stderr, "main: InputBridge::init failed\n");
        return 1;
    }

    // 2. Initialize engine subsystems
    Renderer renderer;
    if (!renderer.init()) {
        fprintf(stderr, "main: Renderer::init failed\n");
        return 1;
    }

    Slicer slicer;
    if (!slicer.init("shaders/slice_compute.glsl")) {
        fprintf(stderr, "main: Slicer::init failed\n");
        return 1;
    }

    Network network;
    if (!network.init(targetIP, targetPort)) {
        fprintf(stderr, "main: Network::init failed\n");
        return 1;
    }

    // 3. Select application
    IApplication* app = nullptr;
    CubeApp      cubeApp;
    HandApp      handApp(network);
    PongApp      pongApp;
    WireframeApp wireframeApp;
    FluidApp     fluidApp;

    if      (strcmp(appName, "cube")      == 0) app = &cubeApp;
    else if (strcmp(appName, "hand")      == 0) app = &handApp;
    else if (strcmp(appName, "pong")      == 0) app = &pongApp;
    else if (strcmp(appName, "wireframe") == 0) app = &wireframeApp;
    else if (strcmp(appName, "fluid")     == 0) app = &fluidApp;
    else {
        fprintf(stderr, "main: unknown app '%s'\n", appName);
        usage(argv[0]);
        return 1;
    }

    app->setup(renderer);
    bool bypassSlicer = app->bypassSlicer();

    // 4. Launch Docker sidecar (hand_tracker.py)
    if (!noDocker) {
        g_dockerPid = launchDocker();
        if (g_dockerPid < 0) {
            fprintf(stderr, "main: WARNING — Docker sidecar failed to launch\n");
        }
    }

    // 5. Allocate slice buffer
    SliceBuffer* sliceBuffer = new SliceBuffer();

    SharedHandData handData = {};

    fprintf(stderr, "main: entering main loop (app=%s)\n", appName);

    // ----------------------------------------------------------------
    // Main loop
    // ----------------------------------------------------------------
    while (g_running) {
        // Read hand data from shared memory (up to 8 retries)
        inputBridge.read(handData, 8);

        app->update(handData);

        if (bypassSlicer) {
            // HandApp sends UDP directly inside draw()
            app->draw(renderer);
        } else {
            // 3D app: render to voxels → slice → send 120 UDP packets
            renderer.clearVoxels();
            app->draw(renderer);

            slicer.sliceAll(renderer.getVoxelTextureID(), *sliceBuffer);

            for (int i = 0; i < SLICE_COUNT; ++i) {
                network.sendSlice((uint8_t)i, &sliceBuffer->data[i][0][0][0]);
                usleep(200);   // 0.2 ms throttle between slices
            }
        }
    }

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------
    fprintf(stderr, "main: shutting down\n");

    app->teardown(renderer);
    delete sliceBuffer;

    if (g_dockerPid > 0) {
        kill(g_dockerPid, SIGTERM);
        waitpid(g_dockerPid, nullptr, 0);
        g_dockerPid = -1;
    }

    network.shutdown();
    slicer.shutdown();
    renderer.shutdown();
    inputBridge.shutdown();

    shm_unlink(HOLOGRAM_SHM_NAME);
    fprintf(stderr, "main: clean exit\n");
    return 0;
}
