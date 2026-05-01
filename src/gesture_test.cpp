// Standalone gesture-detection tester.
// Spawns the same MediaPipe Docker sidecar as hologram, reads landmarks
// from POSIX shm via InputBridge, prints the detected Gesture (and
// per-finger flags + pinch distance) to stdout on every change.
//
// Usage:
//   ./build/gesture_test                   # spawn the docker sidecar
//   ./build/gesture_test --no-docker       # if you launched hand_tracker yourself
//   ./build/gesture_test --period 5        # also print a status line every N s
//
// Don't run while ./build/hologram is running — both would fight over the
// /hologram_hand shared-memory segment.

#include "shared_defs.h"
#include "engine/InputBridge.h"
#include "application/GestureDetector.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libgen.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// -----------------------------------------------------------------------
// Globals for signal handling
// -----------------------------------------------------------------------
static volatile sig_atomic_t g_running   = 1;
static pid_t                 g_dockerPid = -1;

static void sigHandler(int /*sig*/)
{
    g_running = 0;
    if (g_dockerPid > 0) kill(g_dockerPid, SIGTERM);
}

// -----------------------------------------------------------------------
// Find the project root from /proc/self/exe so volume mounts and the
// docker run command resolve correctly regardless of cwd. (Same logic as
// main.cpp::getHologramRoot.)
// -----------------------------------------------------------------------
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

// -----------------------------------------------------------------------
// Spawn the hand-tracker docker container exactly the way hologram does.
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
    if (access("/dev/video0", F_OK) == 0) {
        args.push_back("--device");
        args.push_back("/dev/video0");
        fprintf(stderr, "gesture_test: /dev/video0 found, passing to container\n");
    } else {
        fprintf(stderr, "gesture_test: /dev/video0 not found, container will search at runtime\n");
    }
    args.insert(args.end(), {
        "hand-pose-v5",
        "python3", "/Hologram/python/hand_tracker.py",
        nullptr
    });

    pid_t pid = fork();
    if (pid == 0) {
        execvp("docker", (char* const*)args.data());
        perror("execvp docker");
        _exit(1);
    }
    if (pid < 0) {
        perror("fork");
        return -1;
    }
    fprintf(stderr, "gesture_test: launched docker sidecar (pid %d)\n", pid);
    return pid;
}

// -----------------------------------------------------------------------
// Print one detection line. Stdout for the actual data; stderr stays
// reserved for diagnostics so you can `tee gesture.log` cleanly.
// -----------------------------------------------------------------------
static void printDetection(double tSec, const SharedHandData& hand, Gesture g)
{
    if (!hand.hand_detected) {
        printf("[%7.2fs] hand: not detected\n", tSec);
        fflush(stdout);
        return;
    }
    bool tu = hand.lm_y[4]  < hand.lm_y[2];
    bool iu = hand.lm_y[8]  < hand.lm_y[6];
    bool mu = hand.lm_y[12] < hand.lm_y[10];
    bool ru = hand.lm_y[16] < hand.lm_y[14];
    bool pu = hand.lm_y[20] < hand.lm_y[18];
    float pd = std::hypot(hand.lm_x[4] - hand.lm_x[8],
                          hand.lm_y[4] - hand.lm_y[8]);
    printf("[%7.2fs] gesture=%-13s  T:%d I:%d M:%d R:%d P:%d  pinch=%.3f\n",
           tSec, gestureName(g), tu, iu, mu, ru, pu, pd);
    fflush(stdout);
}

int main(int argc, char* argv[])
{
    bool   noDocker     = false;
    double periodSec    = 0.0;   // 0 = disabled

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-docker") == 0) {
            noDocker = true;
        } else if (std::strcmp(argv[i], "--period") == 0 && i + 1 < argc) {
            periodSec = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                "Usage: %s [--no-docker] [--period SECONDS]\n"
                "  --no-docker      Don't spawn the hand_tracker container.\n"
                "  --period N       Also print a heartbeat line every N seconds.\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "gesture_test: unknown arg '%s'\n", argv[i]);
            return 1;
        }
    }

    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    // Create shared memory before launching the sidecar so the tracker
    // finds it on its first attach attempt.
    InputBridge bridge;
    if (!bridge.init(/*creator=*/true)) {
        fprintf(stderr, "gesture_test: InputBridge::init failed (is /hologram_hand "
                        "still held by another process?)\n");
        return 1;
    }

    if (!noDocker) {
        g_dockerPid = launchDocker();
        if (g_dockerPid < 0) {
            fprintf(stderr, "gesture_test: WARNING — docker sidecar failed to launch\n");
        }
    }

    fprintf(stderr, "gesture_test: ready (Ctrl-C to exit)\n");

    SharedHandData hand{};
    Gesture lastGesture = Gesture::NONE;
    bool    lastDetected = false;
    bool    haveAnyData  = false;

    auto startTime = std::chrono::steady_clock::now();
    auto lastHeartbeat = startTime;

    while (g_running) {
        if (!bridge.read(hand, 8)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        haveAnyData = true;

        Gesture cur = detectGesture(hand);
        bool detected = hand.hand_detected;

        auto now = std::chrono::steady_clock::now();
        double tSec = std::chrono::duration<double>(now - startTime).count();

        bool changed = (cur != lastGesture) || (detected != lastDetected);
        if (changed) {
            printDetection(tSec, hand, cur);
            lastGesture  = cur;
            lastDetected = detected;
            lastHeartbeat = now;
        } else if (periodSec > 0.0) {
            double sinceBeat =
                std::chrono::duration<double>(now - lastHeartbeat).count();
            if (sinceBeat >= periodSec) {
                printDetection(tSec, hand, cur);
                lastHeartbeat = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    if (!haveAnyData) {
        fprintf(stderr,
            "gesture_test: never received any data from hand_tracker. "
            "Is the docker sidecar healthy? Try --no-docker and run it manually.\n");
    }

    fprintf(stderr, "gesture_test: shutting down\n");
    if (g_dockerPid > 0) {
        kill(g_dockerPid, SIGTERM);
        waitpid(g_dockerPid, nullptr, 0);
        g_dockerPid = -1;
    }
    bridge.shutdown();
    return 0;
}
