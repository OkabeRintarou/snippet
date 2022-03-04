#include "amdgpu_context.h"

using namespace amdgpu;

Context::~Context() {
    if (context_handle_ != nullptr) {
        amdgpu_cs_ctx_free(context_handle_);
    }
}
