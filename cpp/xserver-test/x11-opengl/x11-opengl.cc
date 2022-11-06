#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <epoxy/gl.h>
#include <epoxy/egl.h>
#include <gbm.h>
#include <png.h>
#include <X11/Xlib.h>
#include <string>

GLuint program;
Display *display;
Window window;

EGLDisplay egl_display;
EGLSurface egl_surface;
EGLContext egl_context;

static const int TARGET_SIZE = 256;

using namespace std;

static void xinit() {
    assert((display = XOpenDisplay(nullptr)) != nullptr);

    int screen = XDefaultScreen(display);
    Window root = XDefaultRootWindow(display);
    window = XCreateWindow(display, root, 0, 0, TARGET_SIZE, TARGET_SIZE, 0, 
                        XDefaultDepth(display, screen), InputOutput, DefaultVisual(display, screen), 
                        0, nullptr);
    XMapWindow(display, window);
    XFlush(display);
}

static EGLConfig get_config() {
    EGLConfig conf;
    EGLint num_configs;
    const EGLint conf_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE,
    };
    assert(eglChooseConfig(egl_display, conf_attribs, &conf, 1, &num_configs) == EGL_TRUE);
    return conf;
}

static void render_target_init() {
    xinit();

    egl_display = eglGetDisplay((EGLNativeDisplayType)display);
    assert(egl_display != EGL_NO_DISPLAY);

    EGLint major, minor;
    assert(eglInitialize(egl_display, &major, &minor) == EGL_TRUE);

    assert(eglBindAPI(EGL_OPENGL_API) == EGL_TRUE);

    EGLConfig config = get_config();

    egl_surface = eglCreateWindowSurface(egl_display, config, (EGLNativeWindowType)window, nullptr);
    assert(egl_surface != EGL_NO_SURFACE);

    const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE,
    };
    egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, context_attribs);
    assert(egl_context != EGL_NO_CONTEXT);
    assert(eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context) == EGL_TRUE);
}


static const char *vertex_shader_str = 
"#version 330 core\n"
"layout (location = 0) in vec3 pos;\n"
"void main() {\n"
"   gl_Position = vec4(pos, 1.0f);\n"
"}";


static const char *fragment_shader_str = 
"#version 330 core\n"
"void main() {\n"
"   gl_FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";

static GLuint compile_shader(const char *const source, GLenum type) {
    GLuint shader;
    GLint compiled;

    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint info_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_len);
        if (info_len > 1) {
            string info_log;
            info_log.resize(info_len);
            glGetShaderInfoLog(shader, info_len, nullptr, info_log.data());
            fprintf(stderr, "Error compiling shader: \n%s\n", info_log.c_str());
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;

}

static void init_gles() {
    GLint linked;
    GLuint vertex_shader, frag_shader;

    assert((vertex_shader = compile_shader(vertex_shader_str, GL_VERTEX_SHADER)) != 0);
    assert((frag_shader = compile_shader(fragment_shader_str, GL_FRAGMENT_SHADER)) != 0);
    assert((program = glCreateProgram()) != 0);

    glAttachShader(program, vertex_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint info_len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_len);
        if (info_len > 1) {
            string info_log;
            info_log.resize(info_len);
            glGetProgramInfoLog(program, info_len, nullptr, info_log.data());
            fprintf(stderr, "Error linking program:\n%s\n", info_log.c_str());
        }
        glDeleteProgram(program);
        exit(-1);
    }
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glViewport(0, 0, TARGET_SIZE, TARGET_SIZE);

    glUseProgram(program);
}

static void render() {
    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 
        -1.0f, 1.0f, 0.0f, 
        1.0f, 1.0f, 0.0f,
    };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    eglSwapBuffers(egl_display, egl_surface);

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

int main() {
    render_target_init();
    init_gles();
    render();
    sleep(10);
    return 0;
}
