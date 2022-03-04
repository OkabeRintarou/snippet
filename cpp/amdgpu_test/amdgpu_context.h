#pragma once

#include <amdgpu.h>
#include <amdgpu_drm.h>

namespace amdgpu {

class Device;

class Context {
public:
    amdgpu_context_handle raw_handle() { return context_handle_; }
    const amdgpu_context_handle raw_handle() const { return context_handle_; }

    Context(Context &&o) {
        context_handle_ = o.context_handle_;
        o.context_handle_ = nullptr;
    }

    Context& operator=(Context &&o) {
        if (this != &o) {
            context_handle_ = o.context_handle_;
            o.context_handle_ = nullptr;
        }
        return *this;
    }
    ~Context();
private:
    explicit Context(amdgpu_context_handle h) : context_handle_(h) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;


    friend class Device;
private:
    amdgpu_context_handle context_handle_ = nullptr;
};

}
