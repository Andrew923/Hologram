#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <libgen.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <csignal>

#include "../shared_defs.h"

// -----------------------------------------------------------------------
// Derive the Hologram project root from the running executable's path.
// The binary lives at <root>/build/<name>, so go up two levels.
// -----------------------------------------------------------------------
static std::string getHologramRoot()
{
    char exe[4096] = {};
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return "/root/Hologram";   // fallback
    exe[n] = '\0';
    // exe = .../Hologram/build/test_shm_reader
    // dirname once → .../Hologram/build
    // dirname twice → .../Hologram
    char tmp[4096];
    strncpy(tmp, exe, sizeof(tmp));
    char* buildDir  = dirname(tmp);           // <root>/build
    char* projectDir = dirname(buildDir);      // <root>
    return std::string(projectDir);
}

// -----------------------------------------------------------------------
// Signal handling (mirrors main.cpp)
// -----------------------------------------------------------------------
static volatile sig_atomic_t g_running   = 1;
static pid_t                 g_dockerPid = -1;

static void sigHandler(int)
{
    g_running = 0;
    if (g_dockerPid > 0)
        kill(g_dockerPid, SIGTERM);
}

// -----------------------------------------------------------------------
// Docker sidecar launch (identical to main.cpp)
// -----------------------------------------------------------------------
static pid_t launchDocker()
{
    static std::string root     = getHologramRoot();
    static std::string volMount = root + ":/Hologram";

    std::vector<const char*> args = {
        "docker", "run", "--runtime", "nvidia", "--rm", "--network", "host",
        "--ipc=host",
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
        fprintf(stderr, "test_shm_reader: /dev/video0 found, passing to container\n");
    } else {
        fprintf(stderr, "test_shm_reader: /dev/video0 not found, container will search at runtime\n");
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
    fprintf(stderr, "test_shm_reader: launched docker sidecar (pid %d)\n", pid);
    return pid;
}

// -----------------------------------------------------------------------
// Seqlock read (mirrors InputBridge::read)
// -----------------------------------------------------------------------
static bool shmRead(const SharedHandData* shm, SharedHandData& out, int maxRetries = 16)
{
    auto* seqPtr = reinterpret_cast<const std::atomic<uint32_t>*>(&shm->seq);
    for (int i = 0; i < maxRetries; ++i) {
        uint32_t s1 = seqPtr->load(std::memory_order_acquire);
        if (s1 & 1u) continue;   // write in progress

        SharedHandData local;
        local.hand_detected = shm->hand_detected;
        memcpy(local.lm_x, shm->lm_x, sizeof(shm->lm_x));
        memcpy(local.lm_y, shm->lm_y, sizeof(shm->lm_y));
        local.timestamp = shm->timestamp;

        uint32_t s2 = seqPtr->load(std::memory_order_acquire);
        if (s1 == s2) { local.seq = s1; out = local; return true; }
    }
    return false;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main()
{
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    // 1. Create and zero the shared memory (same as main.cpp — must happen
    //    before Docker starts so Python can open it with O_RDWR only)
    int fd = shm_open(HOLOGRAM_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); return 1; }

    if (ftruncate(fd, sizeof(SharedHandData)) < 0) {
        perror("ftruncate"); return 1;
    }

    auto* shm = (SharedHandData*)mmap(nullptr, sizeof(SharedHandData),
                                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm == MAP_FAILED) { perror("mmap"); return 1; }

    memset(shm, 0, sizeof(SharedHandData));
    fprintf(stderr, "test_shm_reader: created %s (%zu bytes)\n",
            HOLOGRAM_SHM_NAME, sizeof(SharedHandData));

    // 2. Launch Docker sidecar
    g_dockerPid = launchDocker();
    if (g_dockerPid < 0) {
        fprintf(stderr, "test_shm_reader: failed to launch docker\n");
        return 1;
    }

    fprintf(stderr, "test_shm_reader: waiting for hand data... (Ctrl-C to stop)\n\n");

    SharedHandData data = {};
    uint32_t lastSeq = 0;   // matches zeroed shm — avoids printing the empty initial state

    // 3. Poll and print — identical cadence to what main.cpp's update() would see
    while (g_running) {
        if (!shmRead(shm, data)) {
            usleep(1000);
            continue;
        }

        // Only print when Python has written a new frame
        if (data.seq == lastSeq) {
            usleep(5000);   // 5 ms poll
            continue;
        }
        lastSeq = data.seq;

        printf("seq=%-6u  hand=%s  ts=%.4f\n",
               data.seq,
               data.hand_detected ? "YES" : "NO ",
               data.timestamp);

        if (data.hand_detected) {
            // Wrist + all five fingertips
            static const int   JOINTS[] = {0,  4,  8,  12,  16,  20};
            static const char* NAMES[]  = {"WRIST    ", "THUMB_TIP",
                                           "INDEX_TIP", "MID_TIP  ",
                                           "RING_TIP ", "PINKY_TIP"};
            for (int i = 0; i < 6; ++i) {
                int j = JOINTS[i];
                printf("  [%2d] %-9s  x=%.4f  y=%.4f\n",
                       j, NAMES[i], data.lm_x[j], data.lm_y[j]);
            }
        }
        printf("\n");
        fflush(stdout);
    }

    // 4. Cleanup (mirrors main.cpp)
    fprintf(stderr, "\ntest_shm_reader: shutting down\n");

    if (g_dockerPid > 0) {
        kill(g_dockerPid, SIGTERM);
        waitpid(g_dockerPid, nullptr, 0);
    }

    munmap(shm, sizeof(SharedHandData));
    close(fd);
    shm_unlink(HOLOGRAM_SHM_NAME);

    fprintf(stderr, "test_shm_reader: done\n");
    return 0;
}
