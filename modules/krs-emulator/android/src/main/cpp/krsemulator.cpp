#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <jni.h>
#include <oboe/Oboe.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "libretro.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "KrsEmulator", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "KrsEmulator", __VA_ARGS__)

// Newer than the libretro.h that ships with mGBA.
#ifndef RETRO_DEVICE_ID_POINTER_IS_OFFSCREEN
#define RETRO_DEVICE_ID_POINTER_IS_OFFSCREEN 15
#endif

namespace {

using Clock = std::chrono::steady_clock;

// Matches the system numbers in KrsEmulatorNative.kt.
enum class System { Gba = 0, Nds = 1, N3ds = 2 };
constexpr int kSystemCount = 3;

System systemFromId(jint id) {
  switch (id) {
    case 1:
      return System::Nds;
    case 2:
      return System::N3ds;
    default:
      return System::Gba;
  }
}

// Systems with a second, touch-sensitive screen below the main one.
bool hasTwoScreens(System system) { return system == System::Nds || system == System::N3ds; }

// The GBA addresses at most 32 MB of ROM; mGBA ignores anything past that, so slightly
// oversized (padded) dumps still load. Beyond this it's surely not a GBA ROM.
constexpr size_t kMaxGbaRomSize = 64 * 1024 * 1024;
// The largest DS cartridges are 512 MB.
constexpr size_t kMaxNdsRomSize = 512 * 1024 * 1024;
constexpr size_t kNdsHeaderSize = 0x200;
constexpr int kSaveCheckFrames = 120;

// --- Cores ----------------------------------------------------------------------------------------

#define KRS_CORE_FUNCTIONS(X)        \
  X(retro_set_environment)           \
  X(retro_set_video_refresh)         \
  X(retro_set_audio_sample)          \
  X(retro_set_audio_sample_batch)    \
  X(retro_set_input_poll)            \
  X(retro_set_input_state)           \
  X(retro_init)                      \
  X(retro_deinit)                    \
  X(retro_api_version)               \
  X(retro_get_system_info)           \
  X(retro_get_system_av_info)        \
  X(retro_reset)                     \
  X(retro_run)                       \
  X(retro_serialize_size)            \
  X(retro_serialize)                 \
  X(retro_unserialize)               \
  X(retro_cheat_reset)               \
  X(retro_cheat_set)                 \
  X(retro_load_game)                 \
  X(retro_unload_game)               \
  X(retro_get_memory_data)           \
  X(retro_get_memory_size)

// A libretro core opened at run time. Every core exports the same retro_* names, so they can't
// all be linked into this library; each is opened privately and called through these pointers.
struct CoreLibrary {
  void *handle = nullptr;
#define KRS_CORE_MEMBER(name) decltype(&::name) name = nullptr;
  KRS_CORE_FUNCTIONS(KRS_CORE_MEMBER)
#undef KRS_CORE_MEMBER
};

const char *coreFileName(System system) {
  switch (system) {
    case System::Nds:
      return "libmelondsds_libretro.so";
    case System::N3ds:
      return "libazahar_libretro.so";
    default:
      return "libmgba_libretro.so";
  }
}

// Opens the system's core the first time it's needed. Cores stay loaded for the life of the app,
// so the returned pointer never dangles. Null if this device's build doesn't include the core.
const CoreLibrary *loadCore(System system) {
  static std::mutex mutex;
  static CoreLibrary cores[kSystemCount];
  std::lock_guard<std::mutex> lock(mutex);
  CoreLibrary &core = cores[static_cast<int>(system)];
  if (core.handle) return &core;

  void *handle = dlopen(coreFileName(system), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    LOGE("Could not open %s: %s", coreFileName(system), dlerror());
    return nullptr;
  }
  CoreLibrary loaded;
  bool complete = true;
#define KRS_CORE_LOAD(name)                                                      \
  loaded.name = reinterpret_cast<decltype(loaded.name)>(dlsym(handle, #name));   \
  if (!loaded.name) {                                                            \
    LOGE("%s is missing %s", coreFileName(system), #name);                       \
    complete = false;                                                            \
  }
  KRS_CORE_FUNCTIONS(KRS_CORE_LOAD)
#undef KRS_CORE_LOAD
  if (!complete) {
    dlclose(handle);
    return nullptr;
  }
  loaded.handle = handle;
  core = loaded;
  return &core;
}

// The running game's core. Set before the emulator thread starts and cleared after it ends.
const CoreLibrary *gCore = nullptr;
System gSystem = System::Gba;

// --- Audio ----------------------------------------------------------------------------------------

// Single-producer (emulator thread), single-consumer (audio callback) stereo ring buffer.
class AudioRing {
 public:
  static constexpr size_t kCapacity = 4096;

  size_t write(const int16_t *samples, size_t frames) {
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t tail = tail_.load(std::memory_order_acquire);
    const size_t count = std::min(frames, kCapacity - (head - tail));
    for (size_t i = 0; i < count; i++) {
      const size_t slot = (head + i) & (kCapacity - 1);
      data_[slot * 2] = samples[i * 2];
      data_[slot * 2 + 1] = samples[i * 2 + 1];
    }
    head_.store(head + count, std::memory_order_release);
    return count;
  }

  size_t read(int16_t *samples, size_t frames) {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    const size_t head = head_.load(std::memory_order_acquire);
    const size_t count = std::min(frames, head - tail);
    for (size_t i = 0; i < count; i++) {
      const size_t slot = (tail + i) & (kCapacity - 1);
      samples[i * 2] = data_[slot * 2];
      samples[i * 2 + 1] = data_[slot * 2 + 1];
    }
    tail_.store(tail + count, std::memory_order_release);
    return count;
  }

  size_t fill() const {
    return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
  }

  // Only safe while the audio stream is stopped.
  void clear() { tail_.store(head_.load()); }

 private:
  int16_t data_[kCapacity * 2]{};
  std::atomic<size_t> head_{0};
  std::atomic<size_t> tail_{0};
};

constexpr size_t kAudioTargetFill = AudioRing::kCapacity / 4;

AudioRing gRing;
std::atomic<bool> gAudioPrimed{false};
std::atomic<uint64_t> gUnderruns{0};
std::atomic<uint64_t> gOverruns{0};

class AudioOut final : public oboe::AudioStreamDataCallback,
                       public oboe::AudioStreamErrorCallback {
 public:
  bool open(int sampleRate) {
    std::lock_guard<std::mutex> lock(mutex_);
    sampleRate_ = sampleRate;
    wanted_ = true;
    return openLocked();
  }

  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_) stream_->requestStart();
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_) stream_->requestStop();
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    wanted_ = false;
    if (stream_) {
      stream_->close();
      stream_.reset();
    }
  }

  oboe::DataCallbackResult onAudioReady(oboe::AudioStream *, void *audioData,
                                        int32_t numFrames) override {
    auto *out = static_cast<int16_t *>(audioData);
    const size_t got = gRing.read(out, static_cast<size_t>(numFrames));
    if (got < static_cast<size_t>(numFrames)) {
      std::memset(out + got * 2, 0, (numFrames - got) * 2 * sizeof(int16_t));
      if (gAudioPrimed.load()) gUnderruns.fetch_add(1);
    }
    return oboe::DataCallbackResult::Continue;
  }

  // The output device changed (for example headphones were unplugged), so reopen on the new one.
  void onErrorAfterClose(oboe::AudioStream *, oboe::Result error) override {
    if (error != oboe::Result::ErrorDisconnected) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (wanted_ && openLocked()) stream_->requestStart();
  }

 private:
  bool openLocked() {
    oboe::AudioStreamBuilder builder;
    builder.setDirection(oboe::Direction::Output)
        ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
        ->setSharingMode(oboe::SharingMode::Shared)
        ->setUsage(oboe::Usage::Game)
        ->setFormat(oboe::AudioFormat::I16)
        ->setFormatConversionAllowed(true)
        ->setChannelCount(oboe::ChannelCount::Stereo)
        ->setChannelConversionAllowed(true)
        ->setSampleRate(sampleRate_)
        ->setSampleRateConversionQuality(oboe::SampleRateConversionQuality::Medium)
        ->setDataCallback(this)
        ->setErrorCallback(this);
    const oboe::Result result = builder.openStream(stream_);
    if (result != oboe::Result::OK) {
      LOGE("Could not open audio stream: %s", oboe::convertToText(result));
      stream_.reset();
      return false;
    }
    return true;
  }

  std::mutex mutex_;
  std::shared_ptr<oboe::AudioStream> stream_;
  int sampleRate_ = 32768;
  bool wanted_ = false;
};

AudioOut gAudio;

// --- Video ----------------------------------------------------------------------------------------

constexpr int kFilterSmooth = 1;
constexpr int kFilterLcd = 2;
constexpr int kFilterScanlines = 3;
constexpr int kAspect4By3 = 1;
constexpr int kAspectStretch = 2;
constexpr int kScalingInteger = 0;

std::atomic<int> gVideoFilter{0};
std::atomic<float> gVideoStrength{0.4f};
std::atomic<int> gVideoAspect{0};
std::atomic<int> gVideoScaling{1};

// Where each of the DS's two screens goes within the surface, in pixels from the top left. The
// core draws them stacked in one frame; index 0 is its top half and 1 its bottom (touch) half. A
// rectangle with no width is hidden. Set from the UI thread, read when drawing.
struct ScreenRect {
  float left = 0;
  float top = 0;
  float width = 0;
  float height = 0;
};

std::mutex gScreenRectsMutex;
ScreenRect gScreenRects[2];
bool gScreenRectsSet = false;
// Draws the touchscreen before the top screen, so the top screen can pop up over it.
bool gTouchScreenFirst = false;

// The pixel format the core asked for. Only the emulator thread touches it.
retro_pixel_format gPixelFormat = RETRO_PIXEL_FORMAT_RGB565;

size_t bytesPerPixel(retro_pixel_format format) {
  return format == RETRO_PIXEL_FORMAT_XRGB8888 ? 4 : 2;
}

// A copy of the latest frame exactly as the core produced it.
struct Frame {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  size_t pitch = 0;
  retro_pixel_format format = RETRO_PIXEL_FORMAT_RGB565;

  bool empty() const { return pixels.empty(); }
};

// A core that draws with OpenGL ES (the 3DS core) renders into a framebuffer this frontend owns
// instead of handing over pixels. Only the emulator thread touches these.
bool gHwRendering = false;
retro_hw_render_callback gHwRender{};
// The size of the latest picture the core drew into the framebuffer; zero before the first.
int gHwFrameWidth = 0;
int gHwFrameHeight = 0;
// The largest picture the core said it may draw, which the framebuffer has to hold.
unsigned gHwMaxWidth = 0;
unsigned gHwMaxHeight = 0;

// Which part of the core's picture a screen is, as fractions of it from the top left.
struct SourceRect {
  float left;
  float top;
  float right;
  float bottom;
};

// The cores stack the two screens in one picture: the DS's are both 256x192, and the 3DS's 320x240
// touchscreen sits centered under its 400x240 top screen (Azahar's default layout, which
// kN3dsVariables keeps). Index 0 is the top screen and 1 the touchscreen.
SourceRect screenSource(System system, int screen) {
  if (screen == 0) return {0, 0, 1, 0.5f};
  if (system == System::N3ds) return {0.1f, 0.5f, 0.9f, 1};
  return {0, 0.5f, 1, 1};
}

GLuint compileShader(GLenum type, const char *source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (ok != GL_TRUE) LOGE("Shader failed to compile");
  return shader;
}

// Owns the GL context, the window surface, and (for hardware-rendering cores) the framebuffer the
// core draws into. The context outlives window surfaces, so a hardware-rendering core never loses
// its GL objects when the app goes to the background or the view is rebuilt. It lives on the
// emulator thread.
class Renderer {
 public:
  // Creates the context, current on an offscreen surface until a window arrives. [es3] is for
  // cores that render with OpenGL ES 3.
  bool create(bool es3) {
    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display_ == EGL_NO_DISPLAY || !eglInitialize(display_, nullptr, nullptr)) return false;
    es3_ = es3;

    const EGLint renderable = es3 ? EGL_OPENGL_ES3_BIT_KHR : EGL_OPENGL_ES2_BIT;
    const EGLint colorBits = es3 ? 8 : 5;
    const EGLint greenBits = es3 ? 8 : 6;
    // The offscreen surface is a pbuffer where the driver allows one alongside window surfaces,
    // and otherwise no surface at all (EGL_KHR_surfaceless_context, which every ES 3 driver has).
    bool pbuffer = true;
    EGLint numConfigs = 0;
    for (const EGLint surfaceType : {EGL_WINDOW_BIT | EGL_PBUFFER_BIT, EGL_WINDOW_BIT}) {
      const EGLint configAttribs[] = {EGL_RENDERABLE_TYPE, renderable, EGL_SURFACE_TYPE, surfaceType,
                                      EGL_RED_SIZE,        colorBits,  EGL_GREEN_SIZE,   greenBits,
                                      EGL_BLUE_SIZE,       colorBits,  EGL_NONE};
      if (eglChooseConfig(display_, configAttribs, &config_, 1, &numConfigs) && numConfigs > 0) {
        pbuffer = (surfaceType & EGL_PBUFFER_BIT) != 0;
        break;
      }
    }
    if (numConfigs < 1) {
      destroy();
      return false;
    }

    const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, es3 ? 3 : 2, EGL_NONE};
    context_ = eglCreateContext(display_, config_, EGL_NO_CONTEXT, contextAttribs);
    if (context_ == EGL_NO_CONTEXT) {
      destroy();
      return false;
    }
    if (pbuffer) {
      const EGLint pbufferAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
      offscreen_ = eglCreatePbufferSurface(display_, config_, pbufferAttribs);
    }
    if (!eglMakeCurrent(display_, offscreen_, offscreen_, context_)) {
      destroy();
      return false;
    }

    program_ = glCreateProgram();
    glAttachShader(program_, compileShader(GL_VERTEX_SHADER, R"(
      attribute vec2 aPosition;
      attribute vec2 aTexCoord;
      varying vec2 vTexCoord;
      void main() {
        vTexCoord = aTexCoord;
        gl_Position = vec4(aPosition, 0.0, 1.0);
      })"));
    glAttachShader(program_, compileShader(GL_FRAGMENT_SHADER, R"(
      precision highp float;
      varying vec2 vTexCoord;
      uniform sampler2D uTexture;
      uniform vec2 uTextureSize;
      uniform int uFilter;
      uniform float uStrength;
      uniform bool uSwapRedBlue;
      void main() {
        vec4 color = texture2D(uTexture, vTexCoord);
        if (uSwapRedBlue) color = color.bgra;
        vec2 cell = fract(vTexCoord * uTextureSize);
        if (uFilter == 2 && (cell.x > 0.75 || cell.y > 0.75)) {
          color.rgb *= 1.0 - uStrength;
        } else if (uFilter == 3 && cell.y > 0.5) {
          color.rgb *= 1.0 - uStrength;
        }
        gl_FragColor = vec4(color.rgb, 1.0);
      })"));
    glBindAttribLocation(program_, 0, "aPosition");
    glBindAttribLocation(program_, 1, "aTexCoord");
    glLinkProgram(program_);
    glUseProgram(program_);
    filterUniform_ = glGetUniformLocation(program_, "uFilter");
    strengthUniform_ = glGetUniformLocation(program_, "uStrength");
    textureSizeUniform_ = glGetUniformLocation(program_, "uTextureSize");
    swapUniform_ = glGetUniformLocation(program_, "uSwapRedBlue");

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    textureWidth_ = 0;
    textureHeight_ = 0;
    return true;
  }

  void destroy() {
    if (display_ != EGL_NO_DISPLAY) {
      if (context_ != EGL_NO_CONTEXT &&
          eglMakeCurrent(display_, offscreen_, offscreen_, context_)) {
        deleteFramebuffer();
      }
      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      if (window_ != EGL_NO_SURFACE) eglDestroySurface(display_, window_);
      if (offscreen_ != EGL_NO_SURFACE) eglDestroySurface(display_, offscreen_);
      if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
      eglTerminate(display_);
    }
    display_ = EGL_NO_DISPLAY;
    window_ = EGL_NO_SURFACE;
    offscreen_ = EGL_NO_SURFACE;
    context_ = EGL_NO_CONTEXT;
    program_ = 0;
    texture_ = 0;
    textureWidth_ = 0;
    textureHeight_ = 0;
    fbo_ = 0;
    fboTexture_ = 0;
    fboDepth_ = 0;
    fboWidth_ = 0;
    fboHeight_ = 0;
  }

  bool hasContext() const { return context_ != EGL_NO_CONTEXT; }
  bool hasWindow() const { return window_ != EGL_NO_SURFACE; }

  // Draws into [window] from now on, or offscreen when it's null. Returns false if the window
  // couldn't be used, in which case drawing stays offscreen.
  bool setWindow(ANativeWindow *window) {
    if (!hasContext()) return false;
    eglMakeCurrent(display_, offscreen_, offscreen_, context_);
    if (window_ != EGL_NO_SURFACE) {
      eglDestroySurface(display_, window_);
      window_ = EGL_NO_SURFACE;
    }
    if (!window) return true;
    window_ = eglCreateWindowSurface(display_, config_, window, nullptr);
    if (window_ == EGL_NO_SURFACE || !eglMakeCurrent(display_, window_, window_, context_)) {
      if (window_ != EGL_NO_SURFACE) eglDestroySurface(display_, window_);
      window_ = EGL_NO_SURFACE;
      eglMakeCurrent(display_, offscreen_, offscreen_, context_);
      return false;
    }
    // The emulator thread sets the pace, so presenting must never block on the display.
    eglSwapInterval(display_, 0);
    return true;
  }

  // Makes sure the framebuffer a hardware-rendering core draws into holds a picture this size.
  bool ensureFramebuffer(int width, int height, bool depth, bool stencil) {
    if (fbo_ && width <= fboWidth_ && height <= fboHeight_) return true;
    width = std::max(width, fboWidth_);
    height = std::max(height, fboHeight_);
    deleteFramebuffer();

    glGenTextures(1, &fboTexture_);
    glBindTexture(GL_TEXTURE_2D, fboTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTexture_, 0);
    if (depth || stencil) {
      glGenRenderbuffers(1, &fboDepth_);
      glBindRenderbuffer(GL_RENDERBUFFER, fboDepth_);
      glRenderbufferStorage(GL_RENDERBUFFER, stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT24,
                            width, height);
      glBindRenderbuffer(GL_RENDERBUFFER, 0);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                                stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT,
                                GL_RENDERBUFFER, fboDepth_);
    }
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      LOGE("The core's framebuffer is incomplete (0x%x)", status);
      deleteFramebuffer();
      return false;
    }
    fboWidth_ = width;
    fboHeight_ = height;
    LOGI("Core framebuffer is %dx%d", width, height);
    return true;
  }

  GLuint framebuffer() const { return fbo_; }

  // Draws a frame a software-rendering core handed over.
  void draw(const Frame &frame) {
    if (!hasWindow() || frame.empty()) return;
    beginDraw();
    glBindTexture(GL_TEXTURE_2D, texture_);
    // OpenGL ES 2 can't skip row padding, so the texture spans the full pitch and the padding is
    // cropped off with texture coordinates. mGBA's 240-pixel rows are 256 pixels apart, for one.
    const bool wide = frame.format == RETRO_PIXEL_FORMAT_XRGB8888;
    const int textureWidth = static_cast<int>(frame.pitch / bytesPerPixel(frame.format));
    const GLenum format = wide ? GL_RGBA : GL_RGB;
    const GLenum type = wide ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT_5_6_5;
    glPixelStorei(GL_UNPACK_ALIGNMENT, wide ? 4 : 2);
    if (textureWidth != textureWidth_ || frame.height != textureHeight_ || wide != textureWide_) {
      glTexImage2D(GL_TEXTURE_2D, 0, format, textureWidth, frame.height, 0, format, type,
                   frame.pixels.data());
      textureWidth_ = textureWidth;
      textureHeight_ = frame.height;
      textureWide_ = wide;
    } else {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, frame.height, format, type,
                      frame.pixels.data());
    }
    // XRGB8888 is B, G, R, X in memory, which OpenGL ES 2 can only take as RGBA.
    present(textureWidth, frame.height, frame.width, frame.height, wide, false);
  }

  // Draws the picture a hardware-rendering core left in the framebuffer.
  void drawHardware(int width, int height) {
    if (!hasWindow() || !fbo_ || width <= 0 || height <= 0) return;
    beginDraw();
    glBindTexture(GL_TEXTURE_2D, fboTexture_);
    // With a bottom-left origin, the core's picture is stored bottom row first.
    present(fboWidth_, fboHeight_, std::min(width, fboWidth_), std::min(height, fboHeight_), false,
            gHwRender.bottom_left_origin);
  }

  // Copies the hardware-rendered picture out as RGBA8888 rows from the top, for screenshots.
  bool readHardware(int width, int height, std::vector<uint8_t> &rgba) {
    if (!fbo_ || width <= 0 || height <= 0) return false;
    width = std::min(width, fboWidth_);
    height = std::min(height, fboHeight_);
    const size_t rowBytes = static_cast<size_t>(width) * 4;
    std::vector<uint8_t> raw(rowBytes * height);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    if (es3_) glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, raw.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    rgba.resize(raw.size());
    for (int row = 0; row < height; row++) {
      uint8_t *out = rgba.data() + rowBytes * row;
      std::memcpy(out, raw.data() + rowBytes * (height - 1 - row), rowBytes);
      for (size_t pixel = 3; pixel < rowBytes; pixel += 4) out[pixel] = 0xFF;
    }
    return true;
  }

 private:
  // Puts the GL state this renderer relies on back in place. A hardware-rendering core shares the
  // context and leaves its own bindings behind.
  void beginDraw() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (es3_) {
      glBindVertexArray(0);
      glBindSampler(0, 0);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
      glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
      glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glUseProgram(program_);
    glActiveTexture(GL_TEXTURE0);
  }

  // Leaves the bindings at their defaults, which is what a core sharing the context expects.
  void endDraw() {
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
  }

  // Draws the bound texture, whose picture is [contentWidth] by [contentHeight] at its corner, to
  // the window. [flipped] is for a picture stored bottom row first.
  void present(int textureWidth, int textureHeight, int contentWidth, int contentHeight,
               bool swapRedBlue, bool flipped) {
    EGLint viewWidth = 0;
    EGLint viewHeight = 0;
    eglQuerySurface(display_, window_, EGL_WIDTH, &viewWidth);
    eglQuerySurface(display_, window_, EGL_HEIGHT, &viewHeight);
    glViewport(0, 0, viewWidth, viewHeight);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const int filter = gVideoFilter.load();
    const GLint sampling = filter == kFilterSmooth ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sampling);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sampling);
    glUniform1i(filterUniform_, filter == kFilterLcd || filter == kFilterScanlines ? filter : 0);
    glUniform1f(strengthUniform_, gVideoStrength.load());
    glUniform1i(swapUniform_, swapRedBlue ? 1 : 0);
    glUniform2f(textureSizeUniform_, static_cast<GLfloat>(textureWidth),
                static_cast<GLfloat>(textureHeight));

    const GLfloat maxU = static_cast<GLfloat>(contentWidth) / textureWidth;
    const GLfloat maxV = static_cast<GLfloat>(contentHeight) / textureHeight;
    // Texture coordinates for a part of the picture given as fractions from its top left.
    const auto u = [&](float fraction) { return fraction * maxU; };
    const auto v = [&](float fraction) { return (flipped ? 1 - fraction : fraction) * maxV; };

    ScreenRect screens[2];
    bool split = false;
    bool touchFirst = false;
    if (hasTwoScreens(gSystem)) {
      std::lock_guard<std::mutex> lock(gScreenRectsMutex);
      split = gScreenRectsSet;
      touchFirst = gTouchScreenFirst;
      screens[0] = gScreenRects[0];
      screens[1] = gScreenRects[1];
    }
    if (split) {
      // Each screen goes into its own rectangle.
      // The later one draws on top, which is how a pop-up screen covers the other.
      for (int step = 0; step < 2; ++step) {
        const int half = touchFirst ? 1 - step : step;
        const ScreenRect &screen = screens[half];
        if (screen.width <= 0 || screen.height <= 0) continue;
        glViewport(static_cast<GLint>(screen.left),
                   static_cast<GLint>(viewHeight - screen.top - screen.height),
                   static_cast<GLsizei>(screen.width), static_cast<GLsizei>(screen.height));
        const SourceRect source = screenSource(gSystem, half);
        drawQuad(u(source.left), u(source.right), v(source.top), v(source.bottom));
      }
    } else {
      applyPictureViewport(viewWidth, viewHeight, contentWidth, contentHeight);
      drawQuad(0, maxU, v(0), v(1));
    }
    endDraw();
    eglSwapBuffers(display_, window_);
  }

  // Draws the texture between the given coordinates over the whole viewport, top edge first.
  static void drawQuad(GLfloat minU, GLfloat maxU, GLfloat topV, GLfloat bottomV) {
    const GLfloat vertices[] = {-1, 1,  minU, topV,    1, 1,  maxU, topV,
                                -1, -1, minU, bottomV, 1, -1, maxU, bottomV};
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), vertices);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), vertices + 2);
    glEnableVertexAttribArray(1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }

  // Sets where in the surface the picture goes for the chosen aspect ratio and scaling.
  static void applyPictureViewport(int surfaceWidth, int surfaceHeight, int frameWidth,
                                   int frameHeight) {
    float width = static_cast<float>(surfaceWidth);
    float height = static_cast<float>(surfaceHeight);
    if (gVideoAspect.load() != kAspectStretch) {
      // 4:3 imitates a TV for GBA games; it would only squash two stacked screens.
      const float aspect = gVideoAspect.load() == kAspect4By3 && gSystem == System::Gba
                               ? 4.0f / 3.0f
                               : static_cast<float>(frameWidth) / frameHeight;
      height = std::min(height, width / aspect);
      if (gVideoScaling.load() == kScalingInteger) {
        height = std::max(1.0f, std::floor(height / frameHeight)) * frameHeight;
      }
      width = height * aspect;
    }
    const float left = std::floor((surfaceWidth - width) / 2);
    const float top = std::floor((surfaceHeight - height) / 2);
    // OpenGL counts from the bottom; the stored rectangle counts from the top like Android does.
    glViewport(static_cast<GLint>(left), static_cast<GLint>(surfaceHeight - top - height),
               static_cast<GLsizei>(width), static_cast<GLsizei>(height));
  }

  void deleteFramebuffer() {
    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    if (fboTexture_) glDeleteTextures(1, &fboTexture_);
    if (fboDepth_) glDeleteRenderbuffers(1, &fboDepth_);
    fbo_ = 0;
    fboTexture_ = 0;
    fboDepth_ = 0;
    fboWidth_ = 0;
    fboHeight_ = 0;
  }

  GLint filterUniform_ = -1;
  GLint strengthUniform_ = -1;
  GLint textureSizeUniform_ = -1;
  GLint swapUniform_ = -1;
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLConfig config_ = nullptr;
  EGLSurface window_ = EGL_NO_SURFACE;
  EGLSurface offscreen_ = EGL_NO_SURFACE;
  EGLContext context_ = EGL_NO_CONTEXT;
  bool es3_ = false;
  GLuint program_ = 0;
  GLuint texture_ = 0;
  int textureWidth_ = 0;
  int textureHeight_ = 0;
  bool textureWide_ = false;
  GLuint fbo_ = 0;
  GLuint fboTexture_ = 0;
  GLuint fboDepth_ = 0;
  int fboWidth_ = 0;
  int fboHeight_ = 0;
};

Renderer gRenderer;

// --- Emulator state -------------------------------------------------------------------------------

std::thread gEmuThread;
std::atomic<bool> gRunning{false};
std::atomic<bool> gQuit{false};
std::atomic<bool> gPaused{false};
std::atomic<bool> gBackgrounded{false};
std::atomic<bool> gFastForward{false};
std::atomic<bool> gPadFastForward{false};
std::atomic<double> gFastForwardSpeed{2.0};
constexpr double kMinFastForwardSpeed = 1.0;
constexpr double kMaxFastForwardSpeed = 10.0;
// Fast-forward runs far more frames than the display can show. Drawing every one would let the
// texture upload and buffer swap, not the core, cap the speed, which DS (whose frames already cost
// the most) felt first. So while fast-forwarding, frames come no closer together than this.
constexpr auto kFastForwardPresentInterval = std::chrono::milliseconds(15);
Clock::time_point gLastPresent;
std::atomic<double> gFrameSeconds{1.0 / 60};

bool isFastForwarding() {
  return gFastForward.load(std::memory_order_relaxed) ||
         gPadFastForward.load(std::memory_order_relaxed);
}
std::atomic<uint32_t> gButtons{0};
std::atomic<uint32_t> gPadButtons{0};

// The touchscreen, packed so a press and its position always change together: bit 32 is set while
// touching, and the low 32 bits hold the libretro x and y coordinates as two int16s.
std::atomic<uint64_t> gPointer{0};

// A controller's analog sticks as four int16s, from the low bits: left x, left y, right x, right y.
std::atomic<uint64_t> gPadAnalog{0};

// The on-screen Circle Pad, as two int16s from the low bits: x, then y.
std::atomic<uint32_t> gTouchStick{0};

// The 3DS core reads the game from a path, so the ROM's file descriptor is kept open while it runs
// and served to it through the VFS under gRomPath. -1 otherwise.
int gRomFd = -1;

void closeRomFd() {
  if (gRomFd >= 0) close(gRomFd);
  gRomFd = -1;
}

// The last message the core showed, which explains why a game wouldn't load. Emulator thread only.
std::string gCoreMessage;

// core.log in the core save directory: the running game's core log. Null when none is open.
std::atomic<FILE *> gCoreLogFile{nullptr};

void closeCoreLog() {
  if (FILE *file = gCoreLogFile.exchange(nullptr)) std::fclose(file);
}

std::mutex gStateMutex;
std::condition_variable gStateChanged;
ANativeWindow *gWindow = nullptr;
bool gWindowDirty = false;

std::atomic<double> gFps{0};
std::atomic<double> gFrameMs{0};
std::atomic<uint64_t> gSaveWrites{0};
std::atomic<uint64_t> gSaveBytes{0};

std::string gSavePath;
std::string gSaveDir;
std::string gSystemDir;
// Set when the ROM's header would make the real GBA BIOS hang, so the core boots the game directly.
bool gSkipBios = false;
Frame gLastFrame;

// Redraws the latest picture, such as after the surface or the screen arrangement changed.
void redraw() {
  if (gHwRendering) {
    gRenderer.drawHardware(gHwFrameWidth, gHwFrameHeight);
  } else {
    gRenderer.draw(gLastFrame);
  }
}

std::mutex gCommandMutex;
std::vector<std::function<void()>> gCommands;
std::vector<uint8_t> gLastSave;
std::vector<uint8_t> gSaveCandidate;

// Writes save memory to disk. Unless forced, a change must be identical across two checks so a
// save the game is still in the middle of writing never replaces the last good file.
void flushSave(bool force) {
  auto *memory = static_cast<const uint8_t *>(gCore->retro_get_memory_data(RETRO_MEMORY_SAVE_RAM));
  const size_t size = gCore->retro_get_memory_size(RETRO_MEMORY_SAVE_RAM);
  if (!memory || size == 0) return;

  if (gLastSave.size() == size && std::memcmp(gLastSave.data(), memory, size) == 0) {
    gSaveCandidate.clear();
    return;
  }
  const bool stable =
      gSaveCandidate.size() == size && std::memcmp(gSaveCandidate.data(), memory, size) == 0;
  if (!force && !stable) {
    gSaveCandidate.assign(memory, memory + size);
    return;
  }

  const std::string tempPath = gSavePath + ".tmp";
  FILE *file = std::fopen(tempPath.c_str(), "wb");
  if (!file) {
    LOGE("Could not open save file for writing");
    return;
  }
  const bool written = std::fwrite(memory, 1, size, file) == size;
  const bool closed = std::fclose(file) == 0;
  if (!written || !closed || std::rename(tempPath.c_str(), gSavePath.c_str()) != 0) {
    LOGE("Could not write save file");
    std::remove(tempPath.c_str());
    return;
  }
  gLastSave.assign(memory, memory + size);
  gSaveCandidate.clear();
  gSaveWrites.fetch_add(1);
  gSaveBytes.store(size);
}

void loadSave() {
  auto *memory = static_cast<uint8_t *>(gCore->retro_get_memory_data(RETRO_MEMORY_SAVE_RAM));
  const size_t size = gCore->retro_get_memory_size(RETRO_MEMORY_SAVE_RAM);
  gSaveCandidate.clear();
  gLastSave.clear();
  if (!memory || size == 0) return;

  if (FILE *file = std::fopen(gSavePath.c_str(), "rb")) {
    const size_t read = std::fread(memory, 1, size, file);
    std::fclose(file);
    gSaveBytes.store(read);
    LOGI("Restored %zu bytes of save data", read);
  }
  gLastSave.assign(memory, memory + size);
}

// --- libretro callbacks ---------------------------------------------------------------------------

struct CoreVariable {
  const char *key;
  const char *value;
};

// melonDS DS settings that suit a phone. Anything not listed keeps the core's own default.
constexpr CoreVariable kMelonDsVariables[] = {
    // The player lays its controls out around one picture with the screens stacked, and maps
    // touches onto the bottom half, so the layout must stay fixed.
    {"melonds_number_of_screen_layouts", "1"},
    {"melonds_screen_layout1", "top-bottom"},
    {"melonds_screen_gap", "0"},
    // Touches come from the real touchscreen, so there's no need for a cursor.
    {"melonds_touch_mode", "touch"},
    {"melonds_show_cursor", "disabled"},
    // Real BIOS and firmware are used when they're in the system folder, and melonDS's own
    // replacements otherwise.
    {"melonds_sysfile_mode", "native"},
    {"melonds_console_mode", "ds"},
};

// Azahar settings that suit this frontend. Anything not listed keeps the core's own default.
constexpr CoreVariable kN3dsVariables[] = {
    // On Android, "auto" picks Vulkan, which this frontend doesn't provide.
    {"citra_graphics_api", "OpenGL"},
    // The renderer cuts the picture into its two screens (see screenSource), so the core has to
    // keep drawing them stacked, top screen first, at the native size.
    {"citra_layout_option", "default"},
    {"citra_swap_screen", "Top"},
    {"citra_resolution_factor", "1"},
    // Touches come from the real touchscreen, and the right stick is the C-stick, never a cursor.
    {"citra_enable_touch_touchscreen", "enabled"},
    {"citra_enable_mouse_touchscreen", "disabled"},
    {"citra_enable_touch_pointer_timeout", "enabled"},
    {"citra_analog_function", "c_stick"},
    // There's no motion sensor interface yet.
    {"citra_enable_motion", "disabled"},
    // Keeps the console's NAND and SD card (and so every game's saves) in the save directory.
    {"citra_use_libretro_save_path", "LibRetro Default"},
    {"citra_use_disk_shader_cache", "enabled"},
    // Speed over exactness, the same trade Azahar's own Android app makes: the JITs and GPU shaders
    // on, high-level audio, and the cheaper shader multiply, which very few games can tell apart.
    {"citra_use_cpu_jit", "enabled"},
    {"citra_use_hw_shader", "enabled"},
    {"citra_use_shader_jit", "enabled"},
    {"citra_shaders_accurate_mul", "disabled"},
    {"citra_audio_emulation", "hle"},
};

const char *coreVariable(const char *key) {
  if (!key) return nullptr;
  if (gSystem == System::Gba) {
    if (gSkipBios && std::strcmp(key, "mgba_skip_bios") == 0) return "ON";
    return nullptr;
  }
  if (gSystem == System::N3ds) {
    for (const auto &variable : kN3dsVariables) {
      if (std::strcmp(key, variable.key) == 0) return variable.value;
    }
    return nullptr;
  }
  for (const auto &variable : kMelonDsVariables) {
    if (std::strcmp(key, variable.key) == 0) return variable.value;
  }
  return nullptr;
}

// --- File access for the 3DS core -----------------------------------------------------------------

// The 3DS core opens the game itself, by path. A file picked through Android's document picker
// can't be opened again by path (not even through /proc/self/fd, which the app may lack permission
// for), so the core's files all go through this VFS, which hands it the descriptor already open.
// Reads and writes go through pread and pwrite with a position per handle, so any number of handles
// can share that descriptor.

}  // namespace

// libretro leaves these for the frontend to define.
struct retro_vfs_file_handle {
  int fd = -1;
  int64_t position = 0;
  std::string path;
};

struct retro_vfs_dir_handle {
  DIR *dir = nullptr;
  std::string path;
  bool includeHidden = false;
  dirent *entry = nullptr;
};

namespace {

// The path the core is given for the game, which this VFS maps to gRomFd.
std::string gRomPath;

bool isRomPath(const char *path) { return gRomFd >= 0 && path && gRomPath == path; }

const char *vfsGetPath(retro_vfs_file_handle *stream) {
  return stream ? stream->path.c_str() : nullptr;
}

retro_vfs_file_handle *vfsOpen(const char *path, unsigned mode, unsigned) {
  if (!path) return nullptr;
  const bool write = (mode & RETRO_VFS_FILE_ACCESS_WRITE) != 0;
  int fd;
  if (isRomPath(path)) {
    if (write) return nullptr;
    fd = fcntl(gRomFd, F_DUPFD_CLOEXEC, 0);
  } else {
    // The same meanings as libretro's own implementation: "rb", "wb", "w+b", or "r+b" with
    // UPDATE_EXISTING.
    int flags = O_RDONLY;
    if (write && (mode & RETRO_VFS_FILE_ACCESS_UPDATE_EXISTING)) {
      flags = O_RDWR;
    } else if (write) {
      flags = ((mode & RETRO_VFS_FILE_ACCESS_READ) ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
    }
    fd = open(path, flags | O_CLOEXEC, 0666);
  }
  if (fd < 0) return nullptr;
  auto *stream = new retro_vfs_file_handle;
  stream->fd = fd;
  stream->path = path;
  return stream;
}

int vfsClose(retro_vfs_file_handle *stream) {
  if (!stream) return -1;
  const int closed = close(stream->fd);
  delete stream;
  return closed == 0 ? 0 : -1;
}

int64_t vfsSize(retro_vfs_file_handle *stream) {
  struct stat info {};
  return stream && fstat(stream->fd, &info) == 0 ? static_cast<int64_t>(info.st_size) : -1;
}

int64_t vfsTell(retro_vfs_file_handle *stream) { return stream ? stream->position : -1; }

int64_t vfsSeek(retro_vfs_file_handle *stream, int64_t offset, int seekPosition) {
  if (!stream) return -1;
  int64_t base = 0;
  if (seekPosition == RETRO_VFS_SEEK_POSITION_CURRENT) {
    base = stream->position;
  } else if (seekPosition == RETRO_VFS_SEEK_POSITION_END) {
    base = vfsSize(stream);
    if (base < 0) return -1;
  } else if (seekPosition != RETRO_VFS_SEEK_POSITION_START) {
    return -1;
  }
  if (base + offset < 0) return -1;
  stream->position = base + offset;
  // The libretro docs promise the new position, but libretro's own implementation (and so
  // RetroArch) returns 0 like fseek, and Azahar passes the result on as fseek's.
  return 0;
}

int64_t vfsRead(retro_vfs_file_handle *stream, void *data, uint64_t length) {
  if (!stream || !data) return -1;
  auto *out = static_cast<uint8_t *>(data);
  uint64_t total = 0;
  while (total < length) {
    const ssize_t got = pread(stream->fd, out + total, length - total, stream->position);
    if (got < 0 && errno == EINTR) continue;
    if (got < 0) return total > 0 ? static_cast<int64_t>(total) : -1;
    if (got == 0) break;
    total += static_cast<uint64_t>(got);
    stream->position += got;
  }
  return static_cast<int64_t>(total);
}

int64_t vfsWrite(retro_vfs_file_handle *stream, const void *data, uint64_t length) {
  if (!stream || !data) return -1;
  const auto *in = static_cast<const uint8_t *>(data);
  uint64_t total = 0;
  while (total < length) {
    const ssize_t put = pwrite(stream->fd, in + total, length - total, stream->position);
    if (put < 0 && errno == EINTR) continue;
    if (put <= 0) return total > 0 ? static_cast<int64_t>(total) : -1;
    total += static_cast<uint64_t>(put);
    stream->position += put;
  }
  return static_cast<int64_t>(total);
}

// Writes go straight to the file, so there's nothing buffered to flush.
int vfsFlush(retro_vfs_file_handle *stream) { return stream ? 0 : -1; }

int vfsRemove(const char *path) { return path && std::remove(path) == 0 ? 0 : -1; }

int vfsRename(const char *oldPath, const char *newPath) {
  return oldPath && newPath && std::rename(oldPath, newPath) == 0 ? 0 : -1;
}

int64_t vfsTruncate(retro_vfs_file_handle *stream, int64_t length) {
  return stream && ftruncate(stream->fd, length) == 0 ? 0 : -1;
}

int vfsStat(const char *path, int32_t *size) {
  if (!path) return 0;
  struct stat info {};
  const bool found = isRomPath(path) ? fstat(gRomFd, &info) == 0 : stat(path, &info) == 0;
  if (!found) return 0;
  // libretro's stat only has room for 32 bits; sizes come from vfsSize instead.
  if (size) *size = static_cast<int32_t>(std::min<int64_t>(info.st_size, INT32_MAX));
  int flags = RETRO_VFS_STAT_IS_VALID;
  if (S_ISDIR(info.st_mode)) flags |= RETRO_VFS_STAT_IS_DIRECTORY;
  if (S_ISCHR(info.st_mode)) flags |= RETRO_VFS_STAT_IS_CHARACTER_SPECIAL;
  return flags;
}

int vfsMkdir(const char *path) {
  if (!path) return -1;
  if (mkdir(path, 0777) == 0) return 0;
  return errno == EEXIST ? -2 : -1;
}

retro_vfs_dir_handle *vfsOpendir(const char *path, bool includeHidden) {
  if (!path) return nullptr;
  DIR *dir = opendir(path);
  if (!dir) return nullptr;
  auto *handle = new retro_vfs_dir_handle;
  handle->dir = dir;
  handle->path = path;
  handle->includeHidden = includeHidden;
  return handle;
}

bool vfsReaddir(retro_vfs_dir_handle *handle) {
  if (!handle) return false;
  while (dirent *entry = readdir(handle->dir)) {
    const char *name = entry->d_name;
    if (std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0) continue;
    if (!handle->includeHidden && name[0] == '.') continue;
    handle->entry = entry;
    return true;
  }
  handle->entry = nullptr;
  return false;
}

const char *vfsDirentGetName(retro_vfs_dir_handle *handle) {
  return handle && handle->entry ? handle->entry->d_name : nullptr;
}

bool vfsDirentIsDir(retro_vfs_dir_handle *handle) {
  if (!handle || !handle->entry) return false;
  if (handle->entry->d_type != DT_UNKNOWN) return handle->entry->d_type == DT_DIR;
  struct stat info {};
  const std::string path = handle->path + "/" + handle->entry->d_name;
  return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

int vfsClosedir(retro_vfs_dir_handle *handle) {
  if (!handle) return -1;
  closedir(handle->dir);
  delete handle;
  return 0;
}

constexpr unsigned kVfsVersion = 3;
retro_vfs_interface gVfs = {
    vfsGetPath, vfsOpen,     vfsClose,   vfsSize,          vfsTell,       vfsSeek,
    vfsRead,    vfsWrite,    vfsFlush,   vfsRemove,        vfsRename,     vfsTruncate,
    vfsStat,    vfsMkdir,    vfsOpendir, vfsReaddir,       vfsDirentGetName, vfsDirentIsDir,
    vfsClosedir,
};

// The framebuffer a hardware-rendering core draws each frame into.
uintptr_t hwCurrentFramebuffer() { return gRenderer.framebuffer(); }

// Looks up GL functions for a hardware-rendering core. eglGetProcAddress only has to know
// extension functions, so core ES 3 functions come from the library itself.
retro_proc_address_t hwProcAddress(const char *name) {
  static void *gles = dlopen("libGLESv3.so", RTLD_NOW | RTLD_LOCAL);
  void *symbol = gles ? dlsym(gles, name) : nullptr;
  if (!symbol) symbol = reinterpret_cast<void *>(eglGetProcAddress(name));
  return reinterpret_cast<retro_proc_address_t>(symbol);
}

// Notes the largest picture the core may draw, so the framebuffer can grow to hold it.
void noteGeometry(const retro_game_geometry &geometry) {
  gHwMaxWidth = std::max({gHwMaxWidth, geometry.base_width, geometry.max_width});
  gHwMaxHeight = std::max({gHwMaxHeight, geometry.base_height, geometry.max_height});
}

void coreLog(retro_log_level level, const char *format, ...) {
  // Debug output from melonDS is far too chatty to keep. Azahar's says why a game won't load, so it
  // goes to the log file (only).
  const bool debug = level == RETRO_LOG_DEBUG;
  if (debug && gSystem != System::N3ds) return;
  char message[1024];
  va_list args;
  va_start(args, format);
  std::vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  const int priority = level == RETRO_LOG_ERROR  ? ANDROID_LOG_ERROR
                       : level == RETRO_LOG_WARN ? ANDROID_LOG_WARN
                                                 : ANDROID_LOG_INFO;
  if (!debug) __android_log_print(priority, "KrsCore", "%s", message);
  // Some phones (Honor's, for one) drop app logs, so the latest game's log is also kept in a file,
  // up to a few megabytes.
  constexpr long kMaxLogBytes = 4 * 1024 * 1024;
  if (FILE *file = gCoreLogFile.load(); file && std::ftell(file) < kMaxLogBytes) {
    std::fprintf(file, "%s", message);
    if (message[0] && message[std::strlen(message) - 1] != '\n') std::fputc('\n', file);
    std::fflush(file);
  }
}

bool environment(unsigned command, void *data) {
  switch (command) {
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: {
      const auto format = *static_cast<const retro_pixel_format *>(data);
      if (format != RETRO_PIXEL_FORMAT_RGB565 && format != RETRO_PIXEL_FORMAT_XRGB8888) {
        return false;
      }
      gPixelFormat = format;
      return true;
    }
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
      *static_cast<const char **>(data) = gSystemDir.c_str();
      return true;
    case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
      *static_cast<const char **>(data) = gSaveDir.c_str();
      return true;
    case RETRO_ENVIRONMENT_GET_VARIABLE: {
      auto *variable = static_cast<retro_variable *>(data);
      variable->value = coreVariable(variable->key);
      return variable->value != nullptr;
    }
    case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
      static_cast<retro_log_callback *>(data)->log = coreLog;
      return true;
    case RETRO_ENVIRONMENT_SET_MESSAGE:
      if (const auto *message = static_cast<const retro_message *>(data); message->msg) {
        coreLog(RETRO_LOG_INFO, "Core message: %s", message->msg);
        gCoreMessage = message->msg;
      }
      return true;
    case RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO: {
      // The picture's size comes with each frame, so only the timing matters here, and the
      // largest size for a hardware-rendering core's framebuffer.
      const auto *av = static_cast<const retro_system_av_info *>(data);
      if (av->timing.fps > 0) gFrameSeconds.store(1.0 / av->timing.fps);
      noteGeometry(av->geometry);
      return true;
    }
    case RETRO_ENVIRONMENT_SET_GEOMETRY:
      noteGeometry(*static_cast<const retro_game_geometry *>(data));
      return true;
    // Only the 3DS core renders with OpenGL ES; the others were built without it.
    case RETRO_ENVIRONMENT_SET_HW_RENDER: {
      if (gSystem != System::N3ds) return false;
      auto *callback = static_cast<retro_hw_render_callback *>(data);
      const bool supported = callback->context_type == RETRO_HW_CONTEXT_OPENGLES2 ||
                             callback->context_type == RETRO_HW_CONTEXT_OPENGLES3 ||
                             (callback->context_type == RETRO_HW_CONTEXT_OPENGLES_VERSION &&
                              callback->version_major <= 3);
      if (!supported) return false;
      callback->get_current_framebuffer = hwCurrentFramebuffer;
      callback->get_proc_address = hwProcAddress;
      gHwRender = *callback;
      gHwRendering = true;
      return true;
    }
    case RETRO_ENVIRONMENT_GET_VFS_INTERFACE: {
      if (gSystem != System::N3ds) return false;
      auto *info = static_cast<retro_vfs_interface_info *>(data);
      if (info->required_interface_version > kVfsVersion) return false;
      info->required_interface_version = kVfsVersion;
      info->iface = &gVfs;
      return true;
    }
    case RETRO_ENVIRONMENT_SET_HW_SHARED_CONTEXT:
      return gSystem == System::N3ds;
    case RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER:
      if (gSystem != System::N3ds) return false;
      *static_cast<unsigned *>(data) = RETRO_HW_CONTEXT_OPENGLES3;
      return true;
    default:
      return false;
  }
}

void videoRefresh(const void *data, unsigned width, unsigned height, size_t pitch) {
  if (data == RETRO_HW_FRAME_BUFFER_VALID) {
    if (!gHwRendering || width == 0 || height == 0) return;
    gHwFrameWidth = static_cast<int>(width);
    gHwFrameHeight = static_cast<int>(height);
    const auto now = Clock::now();
    if (isFastForwarding() && now - gLastPresent < kFastForwardPresentInterval) return;
    gLastPresent = now;
    gRenderer.drawHardware(gHwFrameWidth, gHwFrameHeight);
    return;
  }
  if (!data || width == 0 || height == 0 || pitch < width * bytesPerPixel(gPixelFormat)) return;
  const auto *pixels = static_cast<const uint8_t *>(data);
  gLastFrame.pixels.assign(pixels, pixels + pitch * height);
  gLastFrame.width = static_cast<int>(width);
  gLastFrame.height = static_cast<int>(height);
  gLastFrame.pitch = pitch;
  gLastFrame.format = gPixelFormat;
  if (!gRenderer.hasWindow()) return;
  const auto now = Clock::now();
  if (isFastForwarding() && now - gLastPresent < kFastForwardPresentInterval) return;
  gLastPresent = now;
  gRenderer.draw(gLastFrame);
}

size_t audioBatch(const int16_t *samples, size_t frames) {
  // Audio can't keep up at double speed, so fast-forward runs silent instead of crackling.
  if (isFastForwarding()) return frames;
  const size_t written = gRing.write(samples, frames);
  if (written < frames) gOverruns.fetch_add(1);
  if (gRing.fill() >= kAudioTargetFill) gAudioPrimed.store(true);
  return frames;
}

void audioSample(int16_t left, int16_t right) {
  const int16_t samples[2] = {left, right};
  audioBatch(samples, 1);
}

void inputPoll() {}

// One analog stick axis. The on-screen Circle Pad steers the left stick whenever the controller's
// left stick is centered.
int16_t analogAxis(unsigned index, unsigned id) {
  const uint64_t sticks = gPadAnalog.load(std::memory_order_relaxed);
  const auto value = static_cast<int16_t>((sticks >> ((index * 2 + id) * 16)) & 0xFFFF);
  if (value != 0 || index != RETRO_DEVICE_INDEX_ANALOG_LEFT) return value;
  const uint32_t touch = gTouchStick.load(std::memory_order_relaxed);
  return static_cast<int16_t>((touch >> (id * 16)) & 0xFFFF);
}

int16_t inputState(unsigned port, unsigned device, unsigned index, unsigned id) {
  if (port != 0) return 0;
  if (device == RETRO_DEVICE_JOYPAD) {
    if (id >= 16) return 0;
    const uint32_t held =
        gButtons.load(std::memory_order_relaxed) | gPadButtons.load(std::memory_order_relaxed);
    return static_cast<int16_t>((held >> id) & 1);
  }
  if (device == RETRO_DEVICE_ANALOG) {
    if (index > RETRO_DEVICE_INDEX_ANALOG_RIGHT || id > RETRO_DEVICE_ID_ANALOG_Y) return 0;
    return analogAxis(index, id);
  }
  if (device == RETRO_DEVICE_POINTER && index == 0) {
    const uint64_t pointer = gPointer.load(std::memory_order_relaxed);
    const bool pressed = (pointer >> 32) & 1;
    switch (id) {
      case RETRO_DEVICE_ID_POINTER_X:
        return static_cast<int16_t>(pointer & 0xFFFF);
      case RETRO_DEVICE_ID_POINTER_Y:
        return static_cast<int16_t>((pointer >> 16) & 0xFFFF);
      case RETRO_DEVICE_ID_POINTER_PRESSED:
        return pressed ? 1 : 0;
      case RETRO_DEVICE_ID_POINTER_IS_OFFSCREEN:
        return pressed ? 0 : 1;
      default:
        return 0;
    }
  }
  return 0;
}

// --- Emulator thread ------------------------------------------------------------------------------

// GL state belongs to the emulator thread, so surface changes are applied here.
void applySurfaceChange() {
  std::lock_guard<std::mutex> lock(gStateMutex);
  if (!gWindowDirty) return;
  if (!gRenderer.setWindow(gWindow)) LOGE("Could not draw to the new surface");
  // A new surface starts black, which would blank the picture behind the pause menu.
  redraw();
  gWindowDirty = false;
  gStateChanged.notify_all();
}

bool isHalted() { return gPaused.load() || gBackgrounded.load(); }

void drainCommands() {
  std::vector<std::function<void()>> commands;
  {
    std::lock_guard<std::mutex> lock(gCommandMutex);
    commands.swap(gCommands);
  }
  for (const auto &command : commands) command();
}

// libretro calls must all come from the emulator thread, so other threads queue work for it.
bool runOnEmulatorThread(std::function<bool()> work) {
  if (!gRunning.load()) return false;
  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> result = promise->get_future();
  {
    std::lock_guard<std::mutex> lock(gCommandMutex);
    gCommands.emplace_back([work = std::move(work), promise] { promise->set_value(work()); });
  }
  gStateChanged.notify_all();
  if (result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) return false;
  return result.get();
}

bool writeStateFile(const std::string &path) {
  const size_t size = gCore->retro_serialize_size();
  if (size == 0) return false;
  std::vector<uint8_t> state(size);
  if (!gCore->retro_serialize(state.data(), size)) return false;

  const std::string tempPath = path + ".tmp";
  FILE *file = std::fopen(tempPath.c_str(), "wb");
  if (!file) return false;
  const bool written = std::fwrite(state.data(), 1, size, file) == size;
  const bool closed = std::fclose(file) == 0;
  if (written && closed && std::rename(tempPath.c_str(), path.c_str()) == 0) return true;
  std::remove(tempPath.c_str());
  return false;
}

bool readStateFile(const std::string &path) {
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) return false;
  std::fseek(file, 0, SEEK_END);
  const long size = std::ftell(file);
  std::rewind(file);
  std::vector<uint8_t> state(size > 0 ? static_cast<size_t>(size) : 0);
  const bool read = !state.empty() && std::fread(state.data(), 1, state.size(), file) == state.size();
  std::fclose(file);
  return read && gCore->retro_unserialize(state.data(), state.size());
}

void waitWhilePaused() {
  flushSave(true);
  gAudio.stop();
  while (isHalted() && !gQuit.load()) {
    applySurfaceChange();
    drainCommands();
    std::unique_lock<std::mutex> lock(gStateMutex);
    gStateChanged.wait_for(lock, std::chrono::milliseconds(50));
  }
  gRing.clear();
  gAudioPrimed.store(false);
  gAudio.start();
}

// Tells Android how long each frame may take and how long it really took, so the CPU governor
// clocks up for a demanding core (3DS above all) instead of letting the emulator thread sit on a
// slow, low-clocked core and miss frames. The API arrived in Android 13, so it's looked up at run
// time and does nothing on older versions.
class PerformanceHint {
 public:
  void start(int64_t targetNanos) {
    using GetManager = void *(*)();
    using CreateSession = void *(*)(void *, const int32_t *, size_t, int64_t);
    auto getManager = reinterpret_cast<GetManager>(dlsym(RTLD_DEFAULT, "APerformanceHint_getManager"));
    auto createSession =
        reinterpret_cast<CreateSession>(dlsym(RTLD_DEFAULT, "APerformanceHint_createSession"));
    update_ = reinterpret_cast<UpdateTarget>(
        dlsym(RTLD_DEFAULT, "APerformanceHint_updateTargetWorkDuration"));
    report_ = reinterpret_cast<ReportActual>(
        dlsym(RTLD_DEFAULT, "APerformanceHint_reportActualWorkDuration"));
    close_ = reinterpret_cast<CloseSession>(dlsym(RTLD_DEFAULT, "APerformanceHint_closeSession"));
    if (!getManager || !createSession || !update_ || !report_ || !close_) return;
    void *manager = getManager();
    if (!manager) return;
    const int32_t tid = gettid();
    session_ = createSession(manager, &tid, 1, targetNanos);
    target_ = targetNanos;
  }

  void report(int64_t actualNanos, int64_t targetNanos) {
    if (!session_) return;
    if (targetNanos != target_) {
      update_(session_, targetNanos);
      target_ = targetNanos;
    }
    if (actualNanos > 0) report_(session_, actualNanos);
  }

  void stop() {
    if (session_) close_(session_);
    session_ = nullptr;
  }

 private:
  using UpdateTarget = int (*)(void *, int64_t);
  using ReportActual = int (*)(void *, int64_t);
  using CloseSession = void (*)(void *);
  void *session_ = nullptr;
  int64_t target_ = 0;
  UpdateTarget update_ = nullptr;
  ReportActual report_ = nullptr;
  CloseSession close_ = nullptr;
};

int64_t toNanos(double seconds) { return static_cast<int64_t>(seconds * 1e9); }

// The core's own explanation of why the game didn't start, or [fallback].
std::string coreMessageOr(const char *fallback) {
  return gCoreMessage.empty() ? fallback : gCoreMessage;
}

// [rom] holds the game for cores that take it from memory; [romPath] names it for one that reads
// it itself.
void emulatorMain(std::vector<uint8_t> rom, std::string romPath,
                  std::promise<std::string> result) {
  gPixelFormat = RETRO_PIXEL_FORMAT_RGB565;
  gHwRendering = false;
  gHwRender = {};
  gHwFrameWidth = 0;
  gHwFrameHeight = 0;
  gHwMaxWidth = 0;
  gHwMaxHeight = 0;
  gCoreMessage.clear();
  gCore->retro_set_environment(environment);
  gCore->retro_set_video_refresh(videoRefresh);
  gCore->retro_set_audio_sample(audioSample);
  gCore->retro_set_audio_sample_batch(audioBatch);
  gCore->retro_set_input_poll(inputPoll);
  gCore->retro_set_input_state(inputState);
  gCore->retro_init();
  gLastFrame = Frame();

  retro_game_info game{};
  if (romPath.empty()) {
    game.data = rom.data();
    game.size = rom.size();
  } else {
    game.path = romPath.c_str();
  }
  if (!gCore->retro_load_game(&game)) {
    gCore->retro_deinit();
    gRunning.store(false);
    result.set_value(coreMessageOr("The emulator could not load this ROM."));
    return;
  }
  // The core keeps its own copy, and DS ROMs can be hundreds of megabytes.
  std::vector<uint8_t>().swap(rom);

  retro_system_av_info av{};
  gCore->retro_get_system_av_info(&av);
  noteGeometry(av.geometry);

  // Undoes a load that got as far as the graphics.
  const auto fail = [&result](const std::string &message, bool contextReset) {
    if (contextReset && gHwRender.context_destroy) gHwRender.context_destroy();
    gCore->retro_unload_game();
    gCore->retro_deinit();
    {
      std::lock_guard<std::mutex> lock(gStateMutex);
      gRenderer.destroy();
      gWindowDirty = true;
      gStateChanged.notify_all();
    }
    gRunning.store(false);
    result.set_value(message);
  };

  // The GL context is made up front, even before there's a surface to draw to: a hardware-rendering
  // core can't load the game without one.
  if (!gRenderer.create(gHwRendering)) {
    fail(gHwRendering ? "This device's graphics don't support OpenGL ES 3, which this system needs."
                      : "Could not start the graphics.",
         false);
    return;
  }
  if (gHwRendering) {
    // Every core should say how large its picture gets; if not, the 3DS's native size will do.
    if (gHwMaxWidth == 0 || gHwMaxHeight == 0) {
      gHwMaxWidth = 400;
      gHwMaxHeight = 480;
    }
    if (!gRenderer.ensureFramebuffer(static_cast<int>(gHwMaxWidth),
                                     static_cast<int>(gHwMaxHeight), gHwRender.depth,
                                     gHwRender.stencil)) {
      fail("Could not start the graphics.", false);
      return;
    }
    // Azahar only loads the game here, now that it can make its GL objects. It reports a game that
    // won't load (most often an encrypted one) only with a message.
    gCoreMessage.clear();
    if (gHwRender.context_reset) gHwRender.context_reset();
    if (!gCoreMessage.empty()) {
      fail(gCoreMessage, true);
      return;
    }
  }
  loadSave();

  gFrameSeconds.store(1.0 / av.timing.fps);
  LOGI("Core ready: %.4f fps, %.0f Hz", av.timing.fps, av.timing.sample_rate);

  gRing.clear();
  gAudioPrimed.store(false);
  const bool audioOpen = gAudio.open(static_cast<int>(av.timing.sample_rate));
  if (audioOpen) gAudio.start();
  result.set_value("");

  // A late frame is a stutter, so the emulator thread runs ahead of the app's ordinary work, at
  // the priority Android gives display work (THREAD_PRIORITY_DISPLAY). Failing that is harmless.
  if (setpriority(PRIO_PROCESS, 0, -4) != 0) LOGI("Could not raise the emulator thread's priority");
  PerformanceHint hint;
  hint.start(toNanos(gFrameSeconds.load()));

  auto nextFrame = Clock::now();
  auto windowStart = nextFrame;
  int windowFrames = 0;
  double windowCost = 0;
  int framesSinceSaveCheck = 0;

  while (!gQuit.load()) {
    applySurfaceChange();
    drainCommands();

    if (isHalted()) {
      waitWhilePaused();
      nextFrame = Clock::now();
      windowStart = nextFrame;
      windowFrames = 0;
      windowCost = 0;
      continue;
    }

    const auto frameStart = Clock::now();
    gCore->retro_run();
    // The core may have announced a larger picture during the frame.
    if (gHwRendering) {
      gRenderer.ensureFramebuffer(static_cast<int>(gHwMaxWidth), static_cast<int>(gHwMaxHeight),
                                  gHwRender.depth, gHwRender.stencil);
    }
    const auto frameEnd = Clock::now();
    windowCost += std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
    windowFrames++;
    hint.report(std::chrono::duration_cast<std::chrono::nanoseconds>(frameEnd - frameStart).count(),
                toNanos(isFastForwarding() ? gFrameSeconds.load() / gFastForwardSpeed.load()
                                           : gFrameSeconds.load()));

    if (++framesSinceSaveCheck >= kSaveCheckFrames) {
      framesSinceSaveCheck = 0;
      flushSave(false);
    }

    if (frameEnd - windowStart >= std::chrono::seconds(1)) {
      const double seconds = std::chrono::duration<double>(frameEnd - windowStart).count();
      gFps.store(windowFrames / seconds);
      gFrameMs.store(windowCost / windowFrames);
      windowStart = frameEnd;
      windowFrames = 0;
      windowCost = 0;
    }

    // Nudge the frame rate by up to 1% to keep the audio buffer near its target without crackle.
    const double frameSeconds = gFrameSeconds.load();
    const double fillRatio = static_cast<double>(gRing.fill()) / AudioRing::kCapacity;
    const double correction = 1.0 + 0.01 * (fillRatio - 0.25);
    const double frameTime = isFastForwarding() ? frameSeconds / gFastForwardSpeed.load()
                                               : frameSeconds * correction;
    nextFrame +=
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(frameTime));

    const auto now = Clock::now();
    if (now > nextFrame + std::chrono::milliseconds(100)) {
      nextFrame = now;
    } else {
      std::this_thread::sleep_until(nextFrame);
    }
  }

  hint.stop();
  drainCommands();
  flushSave(true);
  gAudio.close();
  // A hardware-rendering core frees its GL objects first, while the game (and the context) still
  // exist.
  if (gHwRendering && gHwRender.context_destroy) gHwRender.context_destroy();
  gCore->retro_unload_game();
  gCore->retro_deinit();
  {
    std::lock_guard<std::mutex> lock(gStateMutex);
    gRenderer.destroy();
    gWindowDirty = true;
    gStateChanged.notify_all();
  }
  gRunning.store(false);
}

void stopEmulator() {
  gQuit.store(true);
  gStateChanged.notify_all();
  if (gEmuThread.joinable()) gEmuThread.join();
  gQuit.store(false);
  // The core is done with the game, so its file can go.
  closeRomFd();
  closeCoreLog();
}

// --- ROM checks -----------------------------------------------------------------------------------

size_t maxRomSize(System system) {
  return system == System::Nds ? kMaxNdsRomSize : kMaxGbaRomSize;
}

const char *systemName(System system) {
  switch (system) {
    case System::Nds:
      return "DS";
    case System::N3ds:
      return "3DS";
    default:
      return "GBA";
  }
}

// Reads the whole ROM from a file descriptor. It's read here rather than handed over from Kotlin
// so a large DS ROM never has to fit in the Java heap. Returns an error message, or "" on success.
std::string readRom(int fd, System system, std::vector<uint8_t> &rom) {
  const size_t limit = maxRomSize(system);
  struct stat info {};
  if (fstat(fd, &info) == 0 && info.st_size > 0) {
    if (static_cast<uint64_t>(info.st_size) > limit) {
      return std::string("This file is not a valid ") + systemName(system) + " ROM (too large).";
    }
    rom.reserve(static_cast<size_t>(info.st_size));
  }

  // Some document providers hand out pipes, so read until the end instead of trusting the size.
  constexpr size_t kChunk = 1024 * 1024;
  while (true) {
    const size_t used = rom.size();
    if (used >= limit + 1) {
      return std::string("This file is not a valid ") + systemName(system) + " ROM (too large).";
    }
    const size_t want = std::min(kChunk, limit + 1 - used);
    rom.resize(used + want);
    const ssize_t got = read(fd, rom.data() + used, want);
    if (got < 0 && errno == EINTR) {
      rom.resize(used);
      continue;
    }
    if (got < 0) return "Could not read this file.";
    rom.resize(used + static_cast<size_t>(got));
    if (got == 0) break;
  }
  if (rom.size() > limit) {
    return std::string("This file is not a valid ") + systemName(system) + " ROM (too large).";
  }
  return "";
}

// Only rejects files that can't be a ROM at all. The core itself decides whether it can run the
// rest: GBA ROM hacks and translations often ship with a stale header checksum or a cleared fixed
// byte, and mGBA plays those fine (only the real BIOS refuses them, see gbaHeaderIsValid).
std::string validateRom(const std::vector<uint8_t> &rom, System system) {
  if (system == System::Nds) {
    if (rom.size() < kNdsHeaderSize) return "This file is not a valid DS ROM (too small).";
    return "";
  }
  if (rom.size() < 0xC0) return "This file is not a valid GBA ROM (too small).";
  return "";
}

// Catches a 3DS cartridge image (NCSD) that's been cut short, such as by an interrupted copy, which
// Azahar only reports as "Failed to load ROM metadata". Returns an error message, or "".
std::string checkN3dsImage(int fd) {
  constexpr off_t kHeaderOffset = 0x100;
  constexpr uint64_t kMediaUnit = 0x200;
  uint8_t header[0x60] = {};
  if (pread(fd, header, sizeof(header), kHeaderOffset) != static_cast<ssize_t>(sizeof(header))) {
    return "";
  }
  // Other formats (CXI, 3DSX) have no partition table to check.
  if (std::memcmp(header, "NCSD", 4) != 0) return "";
  struct stat info {};
  if (fstat(fd, &info) != 0 || info.st_size <= 0) return "";
  // The partition table starts at 0x120; partition 0, the game itself, has to be all there.
  // (Trimmed dumps only drop padding after the last partition.)
  const auto word = [&header](size_t at) {
    return static_cast<uint64_t>(header[at]) | (static_cast<uint64_t>(header[at + 1]) << 8) |
           (static_cast<uint64_t>(header[at + 2]) << 16) |
           (static_cast<uint64_t>(header[at + 3]) << 24);
  };
  const uint64_t gameEnd = (word(0x20) + word(0x24)) * kMediaUnit;
  const auto size = static_cast<uint64_t>(info.st_size);
  if (size >= gameEnd) return "";
  char message[200];
  std::snprintf(message, sizeof(message),
                "This 3DS game file is incomplete: it has %llu MB of the game's %llu MB. Copy it "
                "to your device again.",
                static_cast<unsigned long long>(size >> 20),
                static_cast<unsigned long long>(gameEnd >> 20));
  return message;
}

// Whether the real GBA BIOS would accept this header. It locks up on a bad one.
bool gbaHeaderIsValid(const std::vector<uint8_t> &rom) {
  if (rom[0xB2] != 0x96) return false;
  int checksum = 0;
  for (size_t i = 0xA0; i <= 0xBC; i++) checksum -= rom[i];
  checksum = (checksum - 0x19) & 0xFF;
  return checksum == rom[0xBD];
}

std::string toString(JNIEnv *env, jstring value) {
  const char *chars = env->GetStringUTFChars(value, nullptr);
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

bool systemInfo(jint system, retro_system_info &info) {
  const CoreLibrary *core = loadCore(systemFromId(system));
  if (!core) return false;
  core->retro_get_system_info(&info);
  return true;
}

// Converts the core's pixels to tightly packed RGBA8888, the layout of an ARGB_8888 Bitmap.
std::vector<uint8_t> toRgba(const Frame &frame) {
  std::vector<uint8_t> rgba(static_cast<size_t>(frame.width) * frame.height * 4);
  uint8_t *out = rgba.data();
  for (int row = 0; row < frame.height; row++) {
    const uint8_t *line = frame.pixels.data() + row * frame.pitch;
    for (int column = 0; column < frame.width; column++) {
      if (frame.format == RETRO_PIXEL_FORMAT_XRGB8888) {
        uint32_t pixel;
        std::memcpy(&pixel, line + column * 4, 4);
        *out++ = (pixel >> 16) & 0xFF;
        *out++ = (pixel >> 8) & 0xFF;
        *out++ = pixel & 0xFF;
      } else {
        uint16_t pixel;
        std::memcpy(&pixel, line + column * 2, 2);
        const uint8_t red = (pixel >> 11) & 0x1F;
        const uint8_t green = (pixel >> 5) & 0x3F;
        const uint8_t blue = pixel & 0x1F;
        *out++ = static_cast<uint8_t>((red << 3) | (red >> 2));
        *out++ = static_cast<uint8_t>((green << 2) | (green >> 4));
        *out++ = static_cast<uint8_t>((blue << 3) | (blue >> 2));
      }
      *out++ = 0xFF;
    }
  }
  return rgba;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_coreName(JNIEnv *env, jobject, jint system) {
  retro_system_info info{};
  return systemInfo(system, info) ? env->NewStringUTF(info.library_name) : nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_coreVersion(JNIEnv *env, jobject, jint system) {
  retro_system_info info{};
  return systemInfo(system, info) ? env->NewStringUTF(info.library_version) : nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_validExtensions(JNIEnv *env, jobject,
                                                                jint system) {
  retro_system_info info{};
  return systemInfo(system, info) ? env->NewStringUTF(info.valid_extensions) : nullptr;
}

extern "C" JNIEXPORT jint JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_libretroApiVersion(JNIEnv *, jobject,
                                                                   jint system) {
  const CoreLibrary *core = loadCore(systemFromId(system));
  return core ? static_cast<jint>(core->retro_api_version()) : 0;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_isAAudioRecommended(JNIEnv *, jobject) {
  return oboe::AudioStreamBuilder::isAAudioRecommended() ? JNI_TRUE : JNI_FALSE;
}

// Reads the ROM from romFd (which stays the caller's to close) and starts it on the system's core.
// Returns null on success, or a message describing why the game could not start.
extern "C" JNIEXPORT jstring JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_start(JNIEnv *env, jobject, jint romFd,
                                                      jint systemId, jstring savePath,
                                                      jstring saveDir, jstring systemDir) {
  const System system = systemFromId(systemId);
  const CoreLibrary *core = loadCore(system);
  if (!core) {
    switch (system) {
      case System::Nds:
        return env->NewStringUTF("DS games need a 64-bit device.");
      case System::N3ds:
        return env->NewStringUTF("3DS games need a 64-bit ARM device.");
      default:
        return env->NewStringUTF("The emulator is missing from this build of the app.");
    }
  }

  std::vector<uint8_t> rom;
  std::string romPath;
  if (system == System::N3ds) {
    // 3DS games run to gigabytes, so the core reads the file itself, through a descriptor of our
    // own that stays open while it runs.
    const std::string invalid = checkN3dsImage(romFd);
    if (!invalid.empty()) return env->NewStringUTF(invalid.c_str());
    const int fd = fcntl(romFd, F_DUPFD_CLOEXEC, 0);
    if (fd < 0) return env->NewStringUTF("Could not read this file.");
    stopEmulator();
    gRomFd = fd;
    // Only the VFS (see vfsOpen) ever opens this path. The extension tells Azahar's loader the
    // type when the contents alone don't, as for CIA files.
    romPath = "/proc/self/fd/" + std::to_string(fd) + ".3ds";
    gRomPath = romPath;
  } else {
    std::string invalid = readRom(romFd, system, rom);
    if (invalid.empty()) invalid = validateRom(rom, system);
    if (!invalid.empty()) return env->NewStringUTF(invalid.c_str());
    stopEmulator();
  }

  gCore = core;
  gSystem = system;
  gSkipBios = system == System::Gba && !gbaHeaderIsValid(rom);
  if (gSkipBios) LOGI("ROM header fails the BIOS check; booting without the BIOS intro");
  gSavePath = toString(env, savePath);
  gSaveDir = toString(env, saveDir);
  gSystemDir = toString(env, systemDir);
  gCoreLogFile.store(std::fopen((gSaveDir + "/core.log").c_str(), "w"));
  gPaused.store(false);
  gBackgrounded.store(false);
  // The on-screen fast-forward toggle isn't reset here: the controls view owns it, and a tap made
  // while a large ROM was still loading would otherwise be dropped with the button left lit.
  gPadFastForward.store(false);
  gButtons.store(0);
  gPadButtons.store(0);
  gPointer.store(0);
  gPadAnalog.store(0);
  gTouchStick.store(0);
  gUnderruns.store(0);
  gOverruns.store(0);
  gSaveWrites.store(0);
  gSaveBytes.store(0);
  gFps.store(0);
  gFrameMs.store(0);
  {
    std::lock_guard<std::mutex> lock(gStateMutex);
    gWindowDirty = true;
  }

  std::promise<std::string> promise;
  std::future<std::string> future = promise.get_future();
  gRunning.store(true);
  gEmuThread =
      std::thread(emulatorMain, std::move(rom), std::move(romPath), std::move(promise));

  const std::string error = future.get();
  if (error.empty()) return nullptr;
  gEmuThread.join();
  closeRomFd();
  closeCoreLog();
  return env->NewStringUTF(error.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_stop(JNIEnv *, jobject) {
  stopEmulator();
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setPaused(JNIEnv *, jobject, jboolean paused) {
  gPaused.store(paused == JNI_TRUE);
  gStateChanged.notify_all();
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setButtons(JNIEnv *, jobject, jint buttons) {
  gButtons.store(static_cast<uint32_t>(buttons), std::memory_order_relaxed);
}

// A touch on the touchscreen, as fractions of its width and height from its top left. Touches
// outside it count as lifting the stylus.
extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setPointer(JNIEnv *, jobject, jboolean pressed,
                                                           jfloat x, jfloat y) {
  // The core stacks the screens with the touchscreen in the bottom half and reads the pointer
  // across the whole frame.
  const SourceRect touchscreen = screenSource(gSystem, 1);
  const float u = touchscreen.left + x * (touchscreen.right - touchscreen.left);
  const float v = touchscreen.top + y * (touchscreen.bottom - touchscreen.top);
  const uint64_t previous = gPointer.load(std::memory_order_relaxed);
  if (pressed != JNI_TRUE || u < 0 || u > 1 || y < 0 || y > 1) {
    // Keep the last position: some games read it again on the frame the stylus lifts.
    gPointer.store(previous & 0xFFFFFFFFull, std::memory_order_relaxed);
    return;
  }
  // libretro spans the picture from -0x7FFF to 0x7FFF on both axes.
  const auto toAxis = [](float position) {
    return static_cast<uint16_t>(static_cast<int16_t>(std::lround((position * 2 - 1) * 0x7FFF)));
  };
  const uint64_t packed =
      (1ull << 32) | (static_cast<uint64_t>(toAxis(v)) << 16) | static_cast<uint64_t>(toAxis(u));
  gPointer.store(packed, std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setPadFastForward(JNIEnv *, jobject,
                                                                  jboolean enabled) {
  gPadFastForward.store(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setPadButtons(JNIEnv *, jobject, jint buttons) {
  gPadButtons.store(static_cast<uint32_t>(buttons), std::memory_order_relaxed);
}

// The on-screen Circle Pad, each axis from -1 to 1 with up and left negative.
extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setTouchStick(JNIEnv *, jobject, jfloat x,
                                                              jfloat y) {
  const auto toAxis = [](float value) {
    if (!std::isfinite(value)) return uint32_t{0};
    const auto axis = static_cast<int16_t>(std::lround(std::clamp(value, -1.0f, 1.0f) * 0x7FFF));
    return static_cast<uint32_t>(static_cast<uint16_t>(axis));
  };
  gTouchStick.store(toAxis(x) | (toAxis(y) << 16), std::memory_order_relaxed);
}

// A controller's sticks, each axis from -1 to 1 with up and left negative.
extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setPadAnalog(JNIEnv *, jobject, jfloat leftX,
                                                             jfloat leftY, jfloat rightX,
                                                             jfloat rightY) {
  const auto toAxis = [](float value) {
    if (!std::isfinite(value)) return uint64_t{0};
    const auto axis = static_cast<int16_t>(std::lround(std::clamp(value, -1.0f, 1.0f) * 0x7FFF));
    return static_cast<uint64_t>(static_cast<uint16_t>(axis));
  };
  gPadAnalog.store(toAxis(leftX) | (toAxis(leftY) << 16) | (toAxis(rightX) << 32) |
                       (toAxis(rightY) << 48),
                   std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setVideoSettings(JNIEnv *, jobject, jint filter,
                                                                 jfloat strength, jint aspect,
                                                                 jint scaling) {
  gVideoFilter.store(filter);
  gVideoStrength.store(strength);
  gVideoAspect.store(aspect);
  gVideoScaling.store(scaling);
  // The game may be paused, so the picture has to be redrawn for the change to show.
  runOnEmulatorThread([] {
    redraw();
    return true;
  });
}

// Where the DS's top and bottom screens go in the surface: left, top, width and height in pixels
// for each, top screen first. A zero width hides that screen. An optional ninth value, when not
// zero, draws the touchscreen first so the top screen goes over it. Null goes back to one picture.
extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setScreenRects(JNIEnv *env, jobject,
                                                               jfloatArray rects) {
  {
    std::lock_guard<std::mutex> lock(gScreenRectsMutex);
    const jsize count = rects != nullptr ? env->GetArrayLength(rects) : 0;
    gScreenRectsSet = count >= 8;
    if (gScreenRectsSet) {
      float values[9] = {};
      env->GetFloatArrayRegion(rects, 0, count >= 9 ? 9 : 8, values);
      gTouchScreenFirst = values[8] != 0;
      for (int screen = 0; screen < 2; ++screen) {
        gScreenRects[screen] = {values[screen * 4], values[screen * 4 + 1],
                                values[screen * 4 + 2], values[screen * 4 + 3]};
      }
    }
  }
  if (!gRunning.load()) return;
  // Called during layout on the UI thread, so the redraw is queued rather than waited for. A
  // paused game wouldn't otherwise show the new arrangement.
  {
    std::lock_guard<std::mutex> lock(gCommandMutex);
    gCommands.emplace_back([] { redraw(); });
  }
  gStateChanged.notify_all();
}

// The current picture as tightly packed RGBA8888, or null if nothing has been drawn yet. Its
// width and height are written to size[0] and size[1].
extern "C" JNIEXPORT jbyteArray JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_getLastFrame(JNIEnv *env, jobject,
                                                             jintArray size) {
  auto frame = std::make_shared<std::vector<uint8_t>>();
  auto dimensions = std::make_shared<std::pair<int, int>>();
  const bool copied = runOnEmulatorThread([frame, dimensions] {
    if (gHwRendering) {
      if (!gRenderer.readHardware(gHwFrameWidth, gHwFrameHeight, *frame)) return false;
      *dimensions = {gHwFrameWidth, gHwFrameHeight};
      return true;
    }
    if (gLastFrame.empty()) return false;
    *frame = toRgba(gLastFrame);
    *dimensions = {gLastFrame.width, gLastFrame.height};
    return true;
  });
  if (!copied || env->GetArrayLength(size) < 2) return nullptr;
  const jint sizeValues[] = {dimensions->first, dimensions->second};
  env->SetIntArrayRegion(size, 0, 2, sizeValues);
  jbyteArray result = env->NewByteArray(static_cast<jsize>(frame->size()));
  env->SetByteArrayRegion(result, 0, static_cast<jsize>(frame->size()),
                          reinterpret_cast<const jbyte *>(frame->data()));
  return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setCheats(JNIEnv *env, jobject,
                                                          jobjectArray lines) {
  if (lines == nullptr) return JNI_FALSE;
  std::vector<std::string> codes;
  const jsize count = env->GetArrayLength(lines);
  codes.reserve(static_cast<size_t>(count));
  for (jsize i = 0; i < count; i++) {
    auto line = static_cast<jstring>(env->GetObjectArrayElement(lines, i));
    if (line == nullptr) continue;
    const char *chars = env->GetStringUTFChars(line, nullptr);
    if (chars != nullptr) {
      // One entry per cheat: up to 256 lines of at most 17 characters, joined with '+'.
      std::string code(chars);
      if (!code.empty() && code.size() <= 256 * 18) codes.push_back(std::move(code));
      env->ReleaseStringUTFChars(line, chars);
    }
    env->DeleteLocalRef(line);
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
      return JNI_FALSE;
    }
  }
  const bool applied = runOnEmulatorThread([codes = std::move(codes)] {
    gCore->retro_cheat_reset();
    for (size_t i = 0; i < codes.size(); i++) {
      gCore->retro_cheat_set(static_cast<unsigned>(i), true, codes[i].c_str());
    }
    return true;
  });
  return applied ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setBackgrounded(JNIEnv *, jobject,
                                                                jboolean backgrounded) {
  gBackgrounded.store(backgrounded == JNI_TRUE);
  gStateChanged.notify_all();
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setFastForwardSpeed(JNIEnv *, jobject,
                                                                    jdouble speed) {
  if (!std::isfinite(speed)) return;
  gFastForwardSpeed.store(std::clamp(speed, kMinFastForwardSpeed, kMaxFastForwardSpeed));
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_saveState(JNIEnv *env, jobject, jstring path) {
  const std::string file = toString(env, path);
  return runOnEmulatorThread([file] { return writeStateFile(file); }) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_loadState(JNIEnv *env, jobject, jstring path) {
  const std::string file = toString(env, path);
  return runOnEmulatorThread([file] { return readStateFile(file); }) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_reset(JNIEnv *, jobject) {
  const bool done = runOnEmulatorThread([] {
    gCore->retro_reset();
    return true;
  });
  return done ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setFastForward(JNIEnv *, jobject,
                                                               jboolean enabled) {
  gFastForward.store(enabled == JNI_TRUE);
}

// Blocks until the emulator thread has let go of the old surface, as Android requires.
extern "C" JNIEXPORT void JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_setSurface(JNIEnv *env, jobject, jobject surface) {
  std::unique_lock<std::mutex> lock(gStateMutex);
  if (gWindow) ANativeWindow_release(gWindow);
  gWindow = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
  gWindowDirty = true;
  gStateChanged.notify_all();
  if (gRunning.load()) {
    gStateChanged.wait_for(lock, std::chrono::seconds(1),
                           [] { return !gWindowDirty || !gRunning.load(); });
  }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_expo_modules_krsemulator_KrsEmulatorNative_getStats(JNIEnv *env, jobject) {
  const jdouble values[] = {gFps.load(),
                            gFrameMs.load(),
                            static_cast<jdouble>(gUnderruns.load()),
                            static_cast<jdouble>(gOverruns.load()),
                            static_cast<jdouble>(gSaveWrites.load()),
                            static_cast<jdouble>(gSaveBytes.load())};
  jdoubleArray result = env->NewDoubleArray(6);
  env->SetDoubleArrayRegion(result, 0, 6, values);
  return result;
}
