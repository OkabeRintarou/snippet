#include <jit/jit.h>
#include <stdio.h>

/**
unsigned int gcd(unsigned int x,unsigned int y) {
        if (x == y) {
                return x;
        } else if (x < y) {
                return gcd(x,y - x);
        } else {
                return gcd(x - y,y);
        }
}
*/

int main() {
  jit_context_t ctx;

  ctx = jit_context_create();
  jit_context_build_start(ctx);

  jit_type_t params[2];
  jit_type_t signature;
  params[0] = jit_type_uint;
  params[1] = jit_type_uint;
  signature =
      jit_type_create_signature(jit_abi_cdecl, jit_type_uint, params, 2, 1);

  jit_function_t function;
  function = jit_function_create(ctx, signature);
  jit_type_free(signature);

  jit_value_t x, y;
  jit_value_t temp1, temp2;
  jit_value_t temp3, temp4;
  jit_label_t label1 = jit_label_undefined, label2 = jit_label_undefined;
  jit_value_t temp_args[2];

  x = jit_value_get_param(function, 0);
  y = jit_value_get_param(function, 1);
  temp1 = jit_insn_eq(function, x, y);
  jit_insn_branch_if_not(function, temp1, &label1);

  // return x
  jit_insn_return(function, x);

  // set label1 at this position
  jit_insn_label(function, &label1);

  temp2 = jit_insn_lt(function, x, y);
  jit_insn_branch_if_not(function, temp2, &label2);

  // return gcd(x,y - x)
  temp_args[0] = x;
  temp_args[1] = jit_insn_sub(function, y, x);
  temp3 = jit_insn_call(function, "gcd", function, 0, temp_args, 2, 0);
  jit_insn_return(function, temp3);

  // set label2 at this position
  jit_insn_label(function, &label2);

  // return gcd(x - y,y)
  temp_args[0] = jit_insn_sub(function, x, y);
  temp_args[1] = y;
  temp4 = jit_insn_call(function, "gcd", function, 0, temp_args, 2, 0);

  jit_insn_return(function, temp4);

  jit_function_compile(function);

  jit_context_build_end(ctx);

  jit_uint arg1, arg2;
  jit_uint result;
  void *args[2];
  arg1 = 27;
  arg2 = 14;
  args[0] = &arg1;
  args[1] = &arg2;
  jit_function_apply(function, args, &result);
  printf("gcd(27,14) = %u\n", (unsigned int)result);
  return 0;
}
