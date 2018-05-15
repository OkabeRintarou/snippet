#include <jit/jit.h>
#include <stdio.h>
/*
Builds and compiles the following funcition

int mul_add(int x,int y,int z) {
        return x * y + z;
}

*/

int compile_mul_add(jit_function_t function) {
  printf("compile_mul_add called\n");
  jit_value_t x, y, z;
  x = jit_value_get_param(function, 0);
  y = jit_value_get_param(function, 1);
  z = jit_value_get_param(function, 2);
  jit_value_t temp1, temp2;
  temp1 = jit_insn_mul(function, x, y);
  temp2 = jit_insn_add(function, temp1, z);
  jit_insn_return(function, temp2);
  return 1;
}

int main() {
  jit_context_t ctx;

  // Create a context to hold the JIT's primary stats
  ctx = jit_context_create();

  // Lock the context while we build and compile the function
  jit_context_build_start(ctx);

  // Build the function signature
  jit_function_t function;
  jit_type_t params[3];
  jit_type_t signature;
  params[0] = jit_type_int;
  params[1] = jit_type_int;
  params[2] = jit_type_int;
  signature =
      jit_type_create_signature(jit_abi_cdecl, jit_type_int, params, 3, 1);

  // Create the function object
  function = jit_function_create(ctx, signature);
  jit_type_free(signature);

  // Make the function recompilable
  jit_function_set_recompilable(function);

  // Compile the function
  jit_function_set_on_demand_compiler(function, compile_mul_add);

  // Unlock the context
  jit_context_build_end(ctx);

  // Execute the function and print the result
  jit_int arg1, arg2, arg3;
  jit_int result;
  void *args[3];
  arg1 = 3;
  arg2 = 5;
  arg3 = 2;
  args[0] = &arg1;
  args[1] = &arg2;
  args[2] = &arg3;
  jit_function_apply(function, args, &result);
  printf("mul_add(3,5,2) = %d\n", (int)result);

  // Execute the function again, to demonstrate that the on-demond compile is
  // not invoked a second time
  arg1 = 13;
  arg2 = 5;
  arg3 = 7;
  args[0] = &arg1;
  args[1] = &arg2;
  args[2] = &arg3;
  jit_function_apply(function, args, &result);
  printf("mul_add(13,5,7) = %d\n", (int)result);

  // Force the function to be recompiled. Normally, we'd use another on-demand
  // compiler with greater optimization capabilities
  jit_context_build_start(ctx);
  jit_function_get_on_demand_compiler(function)(function);
  jit_function_compile(function);
  jit_context_build_end(ctx);

  arg1 = 2;
  arg2 = 8;
  arg3 = -3;
  args[0] = &arg1;
  args[1] = &arg2;
  args[2] = &arg3;
  jit_function_apply(function, args, &result);
  printf("mul_add(2,18,-3) = %d\n", (int)result);
  // Clean up
  jit_context_destroy(ctx);
  return 0;
}
