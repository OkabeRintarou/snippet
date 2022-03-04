#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <amdgpu.h>
#include "result.h"

namespace amdgpu {

class Device;

class BufferObject {
public:

    struct Ptr {
        friend class BufferObject;
    public:
        void *ptr() { return addr_; }
        const void *ptr() const { return addr_; }
        const size_t size() const { return size_; }

        operator bool() const { return addr_ != nullptr; }
        void clear() {
            addr_ = nullptr;
            size_ = 0;
        }
    private:
        Ptr() : addr_(nullptr), size_(0) {}
        Ptr(void *addr, size_t size) : addr_(addr), size_(size) {}

        void *addr_;
        size_t size_;
    };

    template<typename T>
    struct TypedPtr {
    public:
        T& operator[](size_t index) { return ptr_[index]; }
        const T& operator[](size_t index) const { return ptr_[index]; }

        const size_t size() const { return size_; }
        T *ptr() { return ptr_; }
        const T* ptr() const { return ptr_; }
        void clear() {
            ptr_ = nullptr;
            size_ = 0;
        }

        void fill(int c = 0) {
            if (ptr_) {
                memset(ptr_, c, size_);
            }
        }
        operator bool() const { return ptr_ != nullptr; }

        TypedPtr() : ptr_(nullptr), size_(0) {}
        TypedPtr(T *p, size_t s) : ptr_(p), size_(s) {}
    private:
        friend class Ptr;
        T *ptr_;
        size_t size_;
    };

    friend class Device;

    ~BufferObject();

    Result<Ptr, int> mmap();
    void munmap();

    bool is_null() const { return !mapped_ptr_; }

    BufferObject(const BufferObject&) = delete;
    BufferObject& operator=(const BufferObject&) = delete;
   
    BufferObject(BufferObject &&o) {
        device_handle_ = o.device_handle_;
        bo_handle_ = o.bo_handle_;
        va_handle_ = o.va_handle_;
        virtual_mc_base_address_ = o.virtual_mc_base_address_;
        bo_size_ = o.bo_size_;
        bo_align_ = o.bo_align_;
        mapped_ptr_ = o.mapped_ptr_;

        o.device_handle_ = nullptr;
        o.bo_handle_ = nullptr;
        o.va_handle_ = nullptr;
        o.virtual_mc_base_address_ = o.bo_size_ = o.bo_align_ = 0;
        o.mapped_ptr_.clear();
    }

    BufferObject& operator=(BufferObject &&o) {
        if (this != &o) {
            device_handle_ = o.device_handle_;
            bo_handle_ = o.bo_handle_;
            va_handle_ = o.va_handle_;
            virtual_mc_base_address_ = o.virtual_mc_base_address_;
            bo_size_ = o.bo_size_;
            bo_align_ = o.bo_align_;
            mapped_ptr_ = o.mapped_ptr_;

            o.device_handle_ = nullptr;
            o.bo_handle_ = nullptr;
            o.va_handle_ = nullptr;
            o.virtual_mc_base_address_ = o.bo_size_ = o.bo_align_ = 0;
            o.mapped_ptr_.clear();
        }
        return *this;
    }

    uint64_t gpu_address() { return virtual_mc_base_address_; }
    const uint64_t gpu_address() const { return virtual_mc_base_address_; }
    amdgpu_bo_handle bo_handle() { return bo_handle_; }
    const amdgpu_bo_handle bo_handle() const { return bo_handle_; }
private:
    BufferObject(amdgpu_device_handle device_handle, amdgpu_bo_handle bo_handle, 
                uint64_t bo_size, uint64_t bo_align) 
        : device_handle_(device_handle), bo_handle_(bo_handle), 
          bo_size_(bo_size), bo_align_(bo_align) 
    {}
private:
    amdgpu_device_handle device_handle_ = nullptr;
    amdgpu_bo_handle bo_handle_ = nullptr;
    amdgpu_va_handle va_handle_ = nullptr;
    uint64_t virtual_mc_base_address_ = 0;
    uint64_t bo_size_ = 0;
    uint64_t bo_align_ = 0;
    Ptr mapped_ptr_;
};

template<typename T>
BufferObject::TypedPtr<T> to_typed_ptr(BufferObject::Ptr ptr) {
    if (ptr) {
        size_t size = ptr.size() / sizeof(T);
        T *p = static_cast<T*>(ptr.ptr());
        return BufferObject::TypedPtr<T>(p, size);
    }
    return BufferObject::TypedPtr<T>();
}

} // namespace amdgpu
