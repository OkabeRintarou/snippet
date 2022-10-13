#include <unistd.h>
#include "amdgpu_test.h"
#include "amdgpu_bo.h"
#include "amdgpu_device.h"

namespace amdgpu {

Device::Device() {
   fd_ = open_device();
   if (fd_ < 0)
       return;

   int r = amdgpu_device_initialize(fd_, &major_version_, &minor_version_, &dev_);
   if (r) {
       close(fd_);
       fd_ = -1;
       dev_ = nullptr;
   }
}

Device::~Device() {
    if (dev_)
        amdgpu_device_deinitialize(dev_);
    if (fd_ >= 0)
        close(fd_);
}

BufferObject Device::alloc_bo(uint64_t size, uint64_t align, unsigned int heap, uint64_t flags) {
   uint64_t mc_addr;
   void *cpu;
   amdgpu_bo_handle bo_handle;
   amdgpu_va_handle va_handle;
   BufferObject bo;

   int r = amdgpu_bo_alloc_and_map(dev_, size, align, heap, flags, &bo_handle, &cpu, &mc_addr, &va_handle);
   if (r)
       return bo;

   bo.size_ = size;
   bo.mc_address_ = mc_addr;
   bo.cpu_address_ = cpu;
   bo.bo_handle_ = bo_handle;
   bo.va_handle_ = va_handle;

   return bo;
}

}
