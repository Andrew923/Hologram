cmake_minimum_required(VERSION 3.16)
project(hologram LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Jetson GLVND: must be set BEFORE find_package(OpenGL)
set(OpenGL_GL_PREFERENCE GLVND)

# --- Find packages ---
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(GLEW   REQUIRED)
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
)

set(APP_SOURCES
    src/application/CubeApp.cpp
    src/application/HandApp.cpp
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
    ${GLEW_INCLUDE_DIRS}
)

# --- Link libraries ---
target_link_libraries(hologram PRIVATE
    OpenGL::OpenGL
    OpenGL::EGL
    GLEW::GLEW
    Threads::Threads
    rt          # for shm_open / shm_unlink
)

# --- Copy shaders next to binary ---
configure_file(src/shaders/slice_compute.glsl
               ${CMAKE_BINARY_DIR}/shaders/slice_compute.glsl COPYONLY)
configure_file(src/shaders/fluid_sim.glsl
               ${CMAKE_BINARY_DIR}/shaders/fluid_sim.glsl COPYONLY)

# Print build summary
message(STATUS "OpenGL libraries : ${OPENGL_LIBRARIES}")
message(STATUS "EGL library      : ${OpenGL_EGL_LIBRARIES}")
message(STATUS "GLEW library     : ${GLEW_LIBRARIES}")
