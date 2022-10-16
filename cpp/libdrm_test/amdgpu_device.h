#pragma once

#include <cstring>
#include <amdgpu.h>
#include <memory>
#include <string>

namespace amdgpu {

class Device;

struct PosixErrorCode {
    PosixErrorCode() = default;
    ~PosixErrorCode() = default;
    int code = 0;
    const char *message() const {
        return strerror(code);
    }
    bool is_valid() const {
        return code == 0;
    }
};

class BufferObject : private PosixErrorCode {
    friend class Device;
public:
    BufferObject() {
        code = ENOENT;
    }
    ~BufferObject();

    uint64_t size() const { return size_; }
    uint64_t gpu_address() const { return mc_address_; }
    void *cpu_address() const { return cpu_address_; }
    amdgpu_bo_handle handle() const { return bo_handle_; }

    bool is_valid() const { return PosixErrorCode::is_valid(); }
    const char *message() const { return PosixErrorCode::message(); }
private:
    uint64_t size_ = 0;
    uint64_t mc_address_ = 0;
    void *cpu_address_ = nullptr;
    amdgpu_bo_handle bo_handle_ = nullptr;
    amdgpu_va_handle va_handle_ = nullptr;
};

class Context : private PosixErrorCode {
    friend class Device;
public:
    Context() {
        code = ENOENT;
    }
    ~Context();
    Context(const Context&) = delete;
    Context &operator=(const Context &) = delete;

    bool is_valid() const { return PosixErrorCode::is_valid(); }
    const char *message() const { return PosixErrorCode::message(); }

    amdgpu_context_handle handle() const { return ctx_handle_; }
private:
    amdgpu_context_handle ctx_handle_ = nullptr;
    Device *device_ = nullptr;
};

class Device {
public:
    Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool is_valid() const { return fd_ != -1 && dev_ != nullptr; }
    amdgpu_device_handle handle() const { return dev_; }

    bool alloc(const amdgpu_bo_alloc_request &req, BufferObject &bo);
    bool alloc(BufferObject &bo);

    bool alloc(Context &ctx);
private:
    int fd_ = -1;
    amdgpu_device_handle  dev_ = nullptr;
    uint32_t major_version_;
    uint32_t minor_version_;
};
}
