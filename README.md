# Hologram

Jetson-side renderer for a swept-volume persistence-of-vision hologram
display. The Jetson rasterises 3D content into a `128×64×128` RGBA8
voxel volume, slices it into 240 angular slices via a compute shader,
and streams each slice over UDP to the Raspberry Pi receiver
([Andrew923/Hologram_Display](https://github.com/Andrew923/Hologram_Display))
which drives the spinning HUB75 LED panels.

## Hardware

| Component | Role |
|---|---|
| Jetson Orin Nano | Renderer + slicer + UDP sender |
| USB camera (`/dev/video0`) | Hand input (MediaPipe in a Docker sidecar) |
| Raspberry Pi Zero 2 W + Adafruit HUB75 Triple Bonnet + 2× HUB75 panels | Receiver (separate repo) |
| Network link | Direct ethernet, default `10.42.0.x`, Pi at `.168` |

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Dependencies: EGL/OpenGL 4.5, libepoxy, pkg-config, CMake ≥ 3.16.

Produced binaries:
| Binary | Purpose |
|---|---|
| `hologram` | Main renderer + UDP sender |
| `gesture_test` | Headless gesture-detection tester (no GL) |
| `calibrate` | Slicer calibration tool |
| `test_shm_reader` | Inspect the hand-data shared memory |

## Running

```bash
./build/hologram                         # menu app, default Pi at 10.42.0.168
./build/hologram --app fluid             # launch directly into one app
./build/hologram --app cat               # bare-name → loads models/cat.glb
./build/hologram --no-docker             # skip the MediaPipe sidecar
./build/hologram --ip 192.168.1.5        # different Pi address
./build/hologram --timing-log            # CSV per-frame timings to /tmp/
./build/hologram --help                  # full flag list
```

The MediaPipe Docker sidecar (`hand-pose-v5` image, runs
`python/hand_tracker.py`) is launched automatically unless
`--no-docker` is set. It writes hand landmarks into POSIX shared
memory `/hologram_hand`; the C++ side reads via `InputBridge` (see
`shared_defs.h` for the layout).

### Stdin REPL — hot-swap apps without restarting

Once running, type a name and hit Enter to switch apps live:

```
cube       → wireframe cube
wave       → 2D shallow-water sim
fluid      → 3D FLIP particle fluid
flow       → strange-attractor particle flow
cell       → 3D cellular automaton
paint      → finger-trail voxel paint
invaders   → space-invaders shooter
morph      → morphing wireframe sculpture
torus      → torus knot
particles  → MediaPipe-driven particle cloud
hand       → 2D hand passthrough (slicer-bypass)
menu       → carousel menu
corridor / city  → fly-through demos
wireframe        → user-supplied OBJ/GLB via --obj
<name>           → searches models/<name>.glb then .obj
help / ls / list / ?   → reprint app list
quit / exit / q        → clean shutdown
```

`menu` is the gesture-driven entry point; the others are for direct
launch and debugging.

## Apps in detail

The full list of apps with their gestures is in
[`src/application/`](src/application). One-liners:

| App | Description | Gestures |
|---|---|---|
| `menu` | Carousel of all apps. Selected slot at +X (90°). | Pinch advances one slot (smooth glide). FIVE_FINGERS held ~1.5 s launches the centred item. |
| `cube` | Wireframe cube driven by hand position + pinch scale. | Hand position rotates/scales. |
| `fluid` | 3D FLIP/PIC particle sim in an annular bowl, anti-compression buoyancy, hand-tilt gravity, pinch spawns droplets from a 15k reserve. | Tilt with index tip, pinch to drip. |
| `wave` | 2D shallow-water height-map (tripled max wave height vs original). | Index finger pushes the surface; pinch openness sets impulse. |
| `flow` | 8 k particles integrating Lorenz/Aizawa/Halvorsen/Thomas attractors. | Pinch swaps to next attractor. |
| `cell` | 3D cellular automaton (B5,6,7/S5,6,7) at 5 Hz steps. | Pinch drips a random cluster at fingertip. |
| `paint` | 3D voxel paint with age-decaying trail, color-cycles by age. | Hand is always painting; pinch openness controls Y. |
| `invaders` | Space-invaders shooter — pyramid ship on the floor, pinch to fire cyan bullets, falling debris (HP 1/2/3 = green/yellow/red), white powerups (laser pillar / shotgun disc), 1 life. | Hand position moves ship, pinch fires. |
| `morph` | Morphing polyhedron driven by palm position. | Hand position. |
| `wireframe` | Renders any OBJ or GLB model. Vertical-axis-only spin once hand is removed. | Hand direction = rotation axis, pinch = scale. |
| `gesture_test` (separate binary) | Headless CLI gesture tester — prints detected gesture + per-finger flags + pinch distance to stdout. | All gestures. |

All non-menu apps return to the menu via held FIVE_FINGERS (open
splayed palm, ~1.5 s).

## Gestures

`src/application/GestureDetector.h` is a single-header rotation-invariant
classifier. The camera looks straight up so the hand can be at any angle
in the image plane — the detector uses joint angles (cosines of
adjacent bone segments) rather than image-axis comparisons.

| Gesture | Test |
|---|---|
| `PINCH` | Thumb tip ↔ index tip 2D distance < 0.04 (normalised) |
| `FIST` | All five digits curled (joint cos < 0.6 at PIP+DIP / MCP+IP) |
| `THUMBS_UP` | Only thumb extended |
| `POINT` | Only index extended |
| `PEACE` | Index + middle extended, no thumb |
| `ONE..FIVE_FINGERS` | Generic n-finger count |

A finger is "extended" only when both its joints are above
`EXT_COS_THRESHOLD = 0.6` (joint angle < ~53°). The thresholds and
constants are at the top of the header.

### Debugging gestures

```bash
./build/gesture_test                # spawns docker, prints on every detection change
./build/gesture_test --no-docker    # if you launched hand_tracker.py yourself
./build/gesture_test --period 5     # also print a heartbeat line every N seconds
```

Output:
```
[  3.42s] gesture=POINT          T:0 I:1 M:0 R:0 P:0  pinch=0.184
[  4.71s] gesture=FIVE_FINGERS   T:1 I:1 M:1 R:1 P:1  pinch=0.215
[  6.08s] hand: not detected
```

T/I/M/R/P are the per-finger "extended" booleans the detector classifies
on. Pinch is the raw thumb-tip↔index-tip distance.

`gesture_test` and `hologram` both create `/hologram_hand` — don't run
them at the same time.

## File layout

```
src/
  main.cpp                  CLI args, render loop, stdin REPL, docker sidecar
  gesture_test.cpp          Standalone gesture tester
  calibrate.cpp             Slicer calibration tool
  shared_defs.h             SharedHandData layout (mirrored on Pi side)
  engine/
    Renderer.{h,cpp}        EGL context, voxel-texture upload
    Slicer.{h,cpp}          Compute-shader voxel → angular-slice readback
    Network.{h,cpp}         UDP RLE packet sender
    InputBridge.{h,cpp}     POSIX shm reader with seqlock
    JsonConfig.{h,cpp}      menu.json loader
    {Obj,Glb}Loader.{h,cpp} Mesh loaders for the wireframe app
  application/              IApplication subclasses, one per app
    GestureDetector.h       Rotation-invariant gesture classifier
    ReturnToMenuWatcher.h   Held-FIVE_FINGERS exit watcher
    VoxelPaint.h            paintVoxel/Cube/Sphere/3DLine helpers
  shaders/                  GLSL compute shaders (slice_compute + per-app dirs)
config/menu.json            Menu carousel entries
models/                     OBJ/GLB models (loaded via wireframe or bare name)
python/                     MediaPipe sidecar (hand_tracker.py)
build/                      cmake out-of-tree
```

## Latency note

End-to-end input latency is ~500 ms (camera → MediaPipe inference →
UDP → Pi DMA). Apps are designed around this: `cube` and `morph` use
held hand positions with smoothing; `menu` uses pinch-as-discrete
event with a 0.4 s cooldown; `invaders` auto-fires on held pinch. The
display itself runs at ~300 Hz panel refresh on the Pi, so visual
updates are smooth — only input response is lagged.
