#include "util.h"
#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

static int recv_fd() {
  int server_fd, client_fd;
  int server_len, client_len;
  sockaddr_un server_addr, client_addr;
  int fd;
  msghdr msg;
  cmsghdr *cmsg;
  char buf[CMSG_SPACE(sizeof(int))], dummy;

  unlink(SERVER_SOCKET);
  memset(&msg, 0, sizeof(msg));

  server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  server_addr.sun_family = AF_UNIX;
  strcpy(server_addr.sun_path, SERVER_SOCKET);
  server_len = sizeof(server_addr);

  bind(server_fd, (sockaddr *)&server_addr, server_len);
  listen(server_fd, 16);

  printf("Server is waiting for client connect...\n");

  client_len = sizeof(client_addr);
  client_fd = accept(server_fd, (struct sockaddr *)&server_addr,
                     (socklen_t *)&client_len);
  if (client_fd == -1) {
    perror("accept");
    exit(-1);
  }

  printf("Server is waiting for client data...\n");

  memset(buf, 0, sizeof(buf));
  struct iovec io = {.iov_base = &dummy, .iov_len = sizeof(dummy)};

  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  if (recvmsg(client_fd, &msg, 0) < 0) {
    perror("recvmsg");
    exit(-1);
  }

  cmsg = CMSG_FIRSTHDR(&msg);
  fd = *(int *)CMSG_DATA(cmsg);
  printf("Server receive fd %d\n", fd);

  close(client_fd);
  unlink(SERVER_SOCKET);

  return fd;
}

static amdgpu_bo_handle import_bo(int dev_fd, int tex_fd) {
  uint32_t major, minor;
  amdgpu_device_handle device_handle = nullptr;

  if (amdgpu_device_initialize(dev_fd, &major, &minor, &device_handle) != 0) {
    fprintf(stderr, "Fail to initialize amdgpu device\n");
    return nullptr;
  }

  amdgpu_bo_import_result res{};
  if (amdgpu_bo_import(device_handle, amdgpu_bo_handle_type_dma_buf_fd,
                       (uint32_t)tex_fd, &res) != 0) {
    fprintf(stderr, "Fail to import amdgpu bo\n");
    return nullptr;
  }

  return res.buf_handle;
}

static GLFWwindow *init_glfw(amdgpu_bo_handle bo_handle) {
  const unsigned SCR_WIDTH = 256;
  const unsigned SCR_HEIGHT = 256;
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  GLFWwindow *window =
      glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Importer", nullptr, nullptr);
  if (window == nullptr) {
    fprintf(stderr, "Fail to create GLFW window\n");
    return nullptr;
  }

  glfwMakeContextCurrent(window);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Fail to initialize GLEW\n");
    return nullptr;
  }

  void *cpu_ptr = nullptr;
  if (amdgpu_bo_cpu_map(bo_handle, &cpu_ptr) != 0 || cpu_ptr == nullptr) {
    fprintf(stderr, "Fail to map amdgpu bo\n");
    return nullptr;
  }

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, cpu_ptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  return window;
}

static void gl_setup_scene() {
  // Shader source that draws a textures quad
  const char *vertex_shader_source =
      "#version 330 core\n"
      "layout (location = 0) in vec3 aPos;\n"
      "layout (location = 1) in vec2 aTexCoords;\n"

      "out vec2 TexCoords;\n"

      "void main()\n"
      "{\n"
      "   TexCoords = aTexCoords;\n"
      "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
      "}\0";
  const char *fragment_shader_source =
      "#version 330 core\n"
      "out vec4 FragColor;\n"

      "in vec2 TexCoords;\n"

      "uniform sampler2D Texture1;\n"

      "void main()\n"
      "{\n"
      "   FragColor = texture(Texture1, TexCoords);\n"
      "}\0";

  // vertex shader
  int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader);
  // fragment shader
  int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader);
  // link shaders
  int shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  // delete shaders
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  // quad
  float vertices[] = {
      0.5f,  0.5f,  0.0f, 1.0f, 0.0f, // top right
      0.5f,  -0.5f, 0.0f, 1.0f, 1.0f, // bottom right
      -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, // bottom left
      -0.5f, 0.5f,  0.0f, 0.0f, 0.0f  // top left
  };

  unsigned int indices[] = {
      0, 1, 3, // first Triangle
      1, 2, 3  // second Triangle
  };

  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void *)(3 * sizeof(float)));

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);

  // Prebind needed stuff for drawing
  glUseProgram(shader_program);
  glBindVertexArray(VAO);
}

static void render_loop(GLFWwindow *window) {
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

  glViewport(0, 0, width, height);

  while (!glfwWindowShouldClose(window)) {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // draw quad
    // VAO and shader program are already bound from the call to gl_setup_scene
    glActiveTexture(GL_TEXTURE0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}

int main(int argc, char *argv[]) {
  int fd = recv_fd();

  int dev_fd = open_device();
  if (dev_fd < 0) {
    fprintf(stderr, "Fail to open device\n");
    exit(-1);
  }

  amdgpu_bo_handle bo_handle = import_bo(dev_fd, fd);
  if (bo_handle == nullptr) {
    return -1;
  }

  auto window = init_glfw(bo_handle);
  if (window == nullptr) {
    return -1;
  }
  gl_setup_scene();
  render_loop(window);
  return 0;
}
