#include "CellApp.h"
#include "../engine/Renderer.h"
#include <epoxy/gl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

bool CellApp::loadProgram(const char* path, GLuint& outProg)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "CellApp: cannot open shader: %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* src = (char*)malloc(len + 1);
    fread(src, 1, len, f);
    fclose(f);
    src[len] = '\0';

    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, (const GLchar**)&src, nullptr);
    glCompileShader(cs);
    free(src);

    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(cs, sizeof(log), nullptr, log);
        fprintf(stderr, "CellApp: shader compile error (%s):\n%s\n", path, log);
        glDeleteShader(cs);
        return false;
    }

    outProg = glCreateProgram();
    glAttachShader(outProg, cs);
    glLinkProgram(outProg);
    glDeleteShader(cs);

    glGetProgramiv(outProg, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(outProg, sizeof(log), nullptr, log);
        fprintf(stderr, "CellApp: shader link error (%s):\n%s\n", path, log);
        glDeleteProgram(outProg);
        outProg = 0;
        return false;
    }
    return true;
}

static GLuint makeCellTex()
{
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_3D, t);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R8UI,
                   CellApp::GRID_W, CellApp::GRID_H, CellApp::GRID_D);
    glBindTexture(GL_TEXTURE_3D, 0);
    return t;
}

// Dispatch the seed shader against texCur_, optionally clearing all cells
// before painting a random ~50%-density sphere at (gx,gy,gz). Used both for
// initial seeding and for pinch-driven respawns.
void CellApp::seedSphere(int gx, int gy, int gz, float radius, bool clear)
{
    glUseProgram(progSeed_);
    glBindImageTexture(0, texCur_, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R8UI);
    glUniform3i(uSeedCenter_, gx, gy, gz);
    glUniform1f(uSeedRadius_, radius);
    glUniform1ui(uSeedSeed_,  ++seedCounter_);
    glUniform1ui(uSeedClear_, clear ? 1u : 0u);
    glDispatchCompute(GRID_W / 4, GRID_H / 4, GRID_D / 4);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void CellApp::setup(Renderer& /*renderer*/)
{
    menuWatcher_.acknowledge();

    bool ok = true;
    ok &= loadProgram("shaders/cell/step.glsl",   progStep_);
    ok &= loadProgram("shaders/cell/render.glsl", progRender_);
    ok &= loadProgram("shaders/cell/seed.glsl",   progSeed_);
    if (!ok) {
        fprintf(stderr, "CellApp: shader load failed; app disabled\n");
        return;
    }

    uStepBirth_   = glGetUniformLocation(progStep_,   "uBirth");
    uStepSurvive_ = glGetUniformLocation(progStep_,   "uSurvive");
    uSeedCenter_  = glGetUniformLocation(progSeed_,   "uCenter");
    uSeedRadius_  = glGetUniformLocation(progSeed_,   "uRadius");
    uSeedSeed_    = glGetUniformLocation(progSeed_,   "uSeed");
    uSeedClear_   = glGetUniformLocation(progSeed_,   "uClear");
    uRenderColor_ = glGetUniformLocation(progRender_, "uColor");

    texCur_  = makeCellTex();
    texNext_ = makeCellTex();

    // Initial seed: clear + drop a fat sphere at centre.
    seedSphere(GRID_W / 2, GRID_H / 2, GRID_D / 2, 10.0f, /*clear=*/true);

    frameCounter_ = 0;
    seedCounter_  = 0;
    fingerActive_ = false;
}

void CellApp::update(const SharedHandData& hand)
{
    menuWatcher_.update(hand);

    fingerActive_ = hand.hand_detected;
    if (!hand.hand_detected) {
        pinchActive_ = false;
        return;
    }

    float lx = clampf(hand.lm_x[8], 0.0f, 1.0f);
    float ly = clampf(hand.lm_y[8], 0.0f, 1.0f);

    // Map hand to grid coords (X horizontal, Z depth, Y from pinch openness).
    fingerGridX_ = (int)clampf(lx * (float)GRID_W,
                               SEED_RADIUS, GRID_W - SEED_RADIUS);
    fingerGridZ_ = (int)clampf(ly * (float)GRID_D,
                               SEED_RADIUS, GRID_D - SEED_RADIUS);

    float pinchDist = std::hypot(hand.lm_x[4] - hand.lm_x[8],
                                 hand.lm_y[4] - hand.lm_y[8]);
    float t = clampf((pinchDist - 0.03f) / 0.22f, 0.0f, 1.0f);
    fingerGridY_ = (int)clampf(6.0f + t * (float)(GRID_H - 12),
                               SEED_RADIUS, GRID_H - SEED_RADIUS);

    pinchActive_ = (pinchDist < 0.06f);
}

void CellApp::draw(Renderer& renderer)
{
    if (!progStep_ || !progRender_ || !progSeed_) return;

    frameCounter_++;

    // Pinch-driven seed: drip a small cluster into the grid every few frames.
    if (pinchActive_ && (frameCounter_ % SEED_INTERVAL) == 0) {
        seedSphere(fingerGridX_, fingerGridY_, fingerGridZ_,
                   SEED_RADIUS, /*clear=*/false);
    }

    // CA step on the chosen interval; otherwise just re-render last state
    // (renderer.clearVoxels() runs every frame so we always have to repaint).
    if ((frameCounter_ % UPDATE_INTERVAL) == 0) {
        glUseProgram(progStep_);
        glBindImageTexture(0, texCur_,  0, GL_TRUE, 0, GL_READ_ONLY,  GL_R8UI);
        glBindImageTexture(1, texNext_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8UI);
        glUniform1ui(uStepBirth_,   (GLuint)birthMask_);
        glUniform1ui(uStepSurvive_, (GLuint)surviveMask_);
        glDispatchCompute(GRID_W / 4, GRID_H / 4, GRID_D / 4);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        std::swap(texCur_, texNext_);
    }

    // Render: each alive grid cell → 2×2×2 voxel block.
    glUseProgram(progRender_);
    glBindImageTexture(0, texCur_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R8UI);
    glBindImageTexture(1, renderer.getVoxelTextureID(), 0, GL_TRUE, 0,
                       GL_WRITE_ONLY, GL_RGBA8);
    glUniform4f(uRenderColor_, 1.0f, 1.0f, 1.0f, 1.0f);  // white
    glDispatchCompute(GRID_W / 4, GRID_H / 4, GRID_D / 4);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                  | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);
}

void CellApp::teardown(Renderer& /*renderer*/)
{
    if (progStep_)   { glDeleteProgram(progStep_);   progStep_   = 0; }
    if (progRender_) { glDeleteProgram(progRender_); progRender_ = 0; }
    if (progSeed_)   { glDeleteProgram(progSeed_);   progSeed_   = 0; }

    GLuint texs[] = {texCur_, texNext_};
    glDeleteTextures(2, texs);
    texCur_ = texNext_ = 0;
}
