#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <unistd.h>
#include <libgen.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <getopt.h>
#include <time.h>

static inline uint64_t nowUs()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

// Derive project root from running executable (<root>/build/<name> → <root>)
static std::string getHologramRoot()
{
    char exe[4096] = {};
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return "/root/Hologram";
    exe[n] = '\0';
    char tmp[4096];
    strncpy(tmp, exe, sizeof(tmp));
    char* buildDir   = dirname(tmp);
    char* projectDir = dirname(buildDir);
    return std::string(projectDir);
}

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
#include "application/TorusKnotApp.h"
#include "application/ParticleApp.h"
#include "application/MenuApp.h"
#include "application/CorridorApp.h"
#include "application/CityApp.h"

#include <string>

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
    static std::string root     = getHologramRoot();
    static std::string volMount = root + ":/Hologram";

    std::vector<const char*> args = {
        "docker", "run", "--runtime", "nvidia", "--rm", "--network", "host",
        "--ipc=host",          // REQUIRED: shares /dev/shm with host
        "--privileged",
        "-e", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python",
        "-v", "/var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket",
        "-v", "/root/jetson-containers/data:/data",
        "-v", volMount.c_str(),
    };

    // Only map the video device if it exists on the host
    if (access("/dev/video0", F_OK) == 0) {
        args.push_back("--device");
        args.push_back("/dev/video0");
        fprintf(stderr, "main: /dev/video0 found, passing to container\n");
    } else {
        fprintf(stderr, "main: /dev/video0 not found, container will search at runtime\n");
    }

    args.insert(args.end(), {
        "hand-pose-v5",
        "python3", "/Hologram/python/hand_tracker.py",
        nullptr
    });

    pid_t pid = fork();
    if (pid == 0) {
        // Child: exec docker
        execvp("docker", (char* const*)args.data());
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
        "Usage: %s --app <name> --ip <pi_ip> --port <port>\n"
        "  --app         Application: cube, hand, pong, wireframe, fluid,\n"
        "                torus, particles, menu  (default: cube)\n"
        "  --ip          Target IP address of Raspberry Pi (default: 10.42.0.169)\n"
        "  --port        Target UDP port (default: 4210)\n"
        "  --obj         Path to .obj file (for wireframe app)\n"
        "  --no-docker   Skip launching the Docker hand tracker sidecar\n"
        "  --timing-log [path]  Write per-frame timing CSV (omit path for auto-name\n"
        "                       under /tmp/hologram_timing_<timestamp>.csv)\n",
        prog);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Defaults
    char     appName[32]      = "cube";
    char     targetIP[64]     = "10.42.0.169";
    uint16_t targetPort       = 4210;
    bool     noDocker         = false;
    char     objPath[256]     = "";
    char     timingPath[512]  = "";   // empty = timing disabled
    bool     timingAuto       = false; // true = auto-generate filename

    // Parse CLI
    static const struct option longopts[] = {
        {"app",         required_argument, nullptr, 'a'},
        {"ip",          required_argument, nullptr, 'i'},
        {"port",        required_argument, nullptr, 'p'},
        {"obj",         required_argument, nullptr, 'o'},
        {"no-docker",   no_argument,       nullptr, 'n'},
        {"timing-log",  optional_argument, nullptr, 't'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "a:i:p:o:nt::h", longopts, nullptr)) != -1) {
        switch (c) {
            case 'a': strncpy(appName, optarg, sizeof(appName)-1); break;
            case 'i': strncpy(targetIP, optarg, sizeof(targetIP)-1); break;
            case 'p': targetPort = (uint16_t)atoi(optarg); break;
            case 'o': strncpy(objPath, optarg, sizeof(objPath)-1); break;
            case 'n': noDocker = true; break;
            case 't':
                if (optarg && optarg[0]) {
                    strncpy(timingPath, optarg, sizeof(timingPath)-1);
                } else {
                    timingAuto = true;
                }
                break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    // Resolve auto timing log path
    if (timingAuto) {
        time_t now = time(nullptr);
        struct tm* tm = localtime(&now);
        snprintf(timingPath, sizeof(timingPath),
                 "/tmp/hologram_timing_%04d%02d%02d_%02d%02d%02d.csv",
                 tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec);
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

    // 3. Instantiate all application singletons
    CubeApp      cubeApp;
    HandApp      handApp(network);
    PongApp      pongApp;
    WireframeApp wireframeApp;
    FluidApp     fluidApp;
    TorusKnotApp torusApp;
    ParticleApp  particleApp;
    MenuApp      menuApp;
    CorridorApp  corridorApp;
    CityApp      cityApp;

    // If --obj was given on the CLI, pre-load the wireframe model.
    if (objPath[0] != '\0') wireframeApp.setModel(objPath);

    // Resolve a string app name (including "wireframe:<path>") to an
    // IApplication pointer. Returns nullptr if unrecognised.
    auto resolveApp = [&](const std::string& name) -> IApplication* {
        if (name == "cube")         return &cubeApp;
        if (name == "hand")         return &handApp;
        if (name == "pong")         return &pongApp;
        if (name == "fluid")        return &fluidApp;
        if (name == "torus")        return &torusApp;
        if (name == "particles")    return &particleApp;
        if (name == "menu")         return &menuApp;
        if (name == "wireframe")    return &wireframeApp;
        if (name.rfind("wireframe:", 0) == 0) {
            wireframeApp.setModel(name.substr(10));
            return &wireframeApp;
        }
        if (name == "corridor")  return &corridorApp;
        if (name == "city")      return &cityApp;
        return nullptr;
    };

    IApplication* app = resolveApp(appName);
    if (!app) {
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

    // 6. Open timing log if requested
    FILE* timingFp = nullptr;
    if (timingPath[0] != '\0') {
        timingFp = fopen(timingPath, "w");
        if (!timingFp) {
            fprintf(stderr, "main: WARNING — cannot open timing log: %s\n", timingPath);
        } else {
            fprintf(timingFp,
                "frame_id,wall_us,syncReadback_us,update_draw_us,"
                "kickDispatch_us,net_total_us,net_max_slice_us,total_frame_us\n");
            fflush(timingFp);
            fprintf(stderr, "main: timing log → %s\n", timingPath);
        }
    }

    uint64_t frameId = 0;

    fprintf(stderr, "main: entering main loop (app=%s)\n", appName);

    // ----------------------------------------------------------------
    // Swap helper — invoked when requestedApp() returns non-null.
    // Tears down the current app, resolves the target, calls setup, and
    // re-evaluates bypassSlicer. Returns false if the request is
    // unknown (stays on current app).
    // ----------------------------------------------------------------
    auto trySwapApp = [&]() -> bool {
        const char* req = app->requestedApp();
        if (!req) return false;

        std::string name(req);
        IApplication* next = resolveApp(name);
        if (!next) {
            fprintf(stderr, "main: swap to unknown '%s' ignored\n", req);
            return false;
        }

        fprintf(stderr, "main: swapping → %s\n", name.c_str());
        app->teardown(renderer);
        app = next;
        app->setup(renderer);
        frameId = 0;
        bool newBypass = app->bypassSlicer();
        bool changed   = (newBypass != bypassSlicer);
        bypassSlicer   = newBypass;
        return changed;   // true if the caller must break and re-enter
    };

    // ----------------------------------------------------------------
    // Main loop — outer `while` re-enters the correct path whenever the
    // bypass flag changes due to an app swap.
    // ----------------------------------------------------------------
    while (g_running) {
      if (bypassSlicer) {
        // HandApp-style path: app sends UDP directly inside draw().
        while (g_running) {
            uint64_t tStart = nowUs();
            inputBridge.read(handData, 8);
            app->update(handData);
            uint64_t tAfterUpdate = nowUs();
            app->draw(renderer);
            uint64_t tEnd = nowUs();

            if (timingFp) {
                fprintf(timingFp, "%llu,%llu,0,%llu,0,%llu,0,%llu\n",
                    (unsigned long long)frameId++,
                    (unsigned long long)tStart,
                    (unsigned long long)(tAfterUpdate - tStart),
                    (unsigned long long)(tEnd - tAfterUpdate),
                    (unsigned long long)(tEnd - tStart));
                if (frameId % 60 == 0) fflush(timingFp);
            }

            if (trySwapApp()) break;  // bypass flag changed → re-enter
        }
      } else {
        // 3D app path: pipeline GPU compute with UDP sending.
        //
        // Prime: render frame 0 and kick GPU before entering loop
        inputBridge.read(handData, 8);
        app->update(handData);
        renderer.clearVoxels();
        app->draw(renderer);
        slicer.kickDispatch(renderer.getVoxelTextureID());

        while (g_running) {
            uint64_t tFrameStart = nowUs();

            slicer.syncReadback(*sliceBuffer);
            uint64_t tAfterSync = nowUs();

            inputBridge.read(handData, 8);
            app->update(handData);
            renderer.clearVoxels();
            app->draw(renderer);
            uint64_t tAfterDraw = nowUs();

            slicer.kickDispatch(renderer.getVoxelTextureID());
            uint64_t tAfterKick = nowUs();

            uint64_t netMax = 0;
            for (int i = 0; i < SLICE_COUNT; ++i) {
                uint64_t ts = nowUs();
                network.sendSlice((uint8_t)i, &sliceBuffer->data[i][0][0][0]);
                usleep(100);
                uint64_t elapsed = nowUs() - ts;
                if (elapsed > netMax) netMax = elapsed;
            }
            uint64_t tAfterNet = nowUs();

            if (timingFp) {
                fprintf(timingFp, "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                    (unsigned long long)frameId++,
                    (unsigned long long)tFrameStart,
                    (unsigned long long)(tAfterSync - tFrameStart),
                    (unsigned long long)(tAfterDraw - tAfterSync),
                    (unsigned long long)(tAfterKick - tAfterDraw),
                    (unsigned long long)(tAfterNet  - tAfterKick),
                    (unsigned long long)netMax,
                    (unsigned long long)(tAfterNet  - tFrameStart));
                if (frameId % 60 == 0) fflush(timingFp);
            }

            if (trySwapApp()) {
                // Drain the in-flight GPU work before switching loop paths.
                slicer.syncReadback(*sliceBuffer);
                break;
            }
        }

        // Drain if we're exiting (g_running==false) rather than swapping.
        if (!g_running) slicer.syncReadback(*sliceBuffer);
      }
    }

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------
    fprintf(stderr, "main: shutting down\n");

    if (timingFp) { fflush(timingFp); fclose(timingFp); timingFp = nullptr; }

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
