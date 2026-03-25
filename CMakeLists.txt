cmake_minimum_required(VERSION 3.16)
project(hologram LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Jetson GLVND: must be set BEFORE find_package(OpenGL)
set(OpenGL_GL_PREFERENCE GLVND)

# --- Find packages ---
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(PkgConfig REQUIRED)
pkg_check_modules(epoxy REQUIRED IMPORTED_TARGET epoxy)
find_package(Threads REQUIRED)

# --- Compiler flags ---
add_compile_options(
    -O2
    -march=armv8-a
    -mtune=cortex-a78
    -Wall
    -Wextra
)

# --- Source files ---
set(ENGINE_SOURCES
    src/engine/Renderer.cpp
    src/engine/Slicer.cpp
    src/engine/Network.cpp
    src/engine/InputBridge.cpp
    src/engine/ObjLoader.cpp
)

set(APP_SOURCES
    src/application/CubeApp.cpp
    src/application/HandApp.cpp
    src/application/WireframeApp.cpp
)

add_executable(hologram
    src/main.cpp
    ${ENGINE_SOURCES}
    ${APP_SOURCES}
)

# --- Include directories ---
target_include_directories(hologram PRIVATE
    ${CMAKE_SOURCE_DIR}          # for shared_defs.h
    ${OPENGL_INCLUDE_DIR}
    ${epoxy_INCLUDE_DIRS}
)

# --- Link libraries ---
target_link_libraries(hologram PRIVATE
    OpenGL::OpenGL
    OpenGL::EGL
    PkgConfig::epoxy
    Threads::Threads
    rt          # for shm_open / shm_unlink
)

# --- test_shm_reader: standalone shared-memory monitor (no GL deps) ---
add_executable(test_shm_reader src/test_shm_reader.cpp)
target_include_directories(test_shm_reader PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_shm_reader PRIVATE rt)

# --- Copy shaders next to binary ---
configure_file(src/shaders/slice_compute.glsl
               ${CMAKE_BINARY_DIR}/shaders/slice_compute.glsl COPYONLY)
configure_file(src/shaders/fluid_sim.glsl
               ${CMAKE_BINARY_DIR}/shaders/fluid_sim.glsl COPYONLY)

# Print build summary
message(STATUS "OpenGL libraries : ${OPENGL_LIBRARIES}")
message(STATUS "EGL library      : ${OpenGL_EGL_LIBRARIES}")
message(STATUS "epoxy library    : ${epoxy_LIBRARIES}")
