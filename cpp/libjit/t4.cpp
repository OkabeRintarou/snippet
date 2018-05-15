#include <cstdio>
#include <jit/jit-plus.h>

class mul_add_function : public jit_function {
public:
	mul_add_function(jit_context &ctx):jit_function(ctx) {
		create();
		set_recompilable();
	}

	virtual void build();
protected:
	virtual jit_type_t create_signature();
};

jit_type_t mul_add_function::create_signature() {
	// Return type, followed by three parameters,
	// terminated with "end_params".
	return signature_helper(
		jit_type_int,jit_type_int,jit_type_int,jit_type_int,end_params);
}

void mul_add_function::build() {
	printf("Compiling mul_add on demand\n");

	jit_value x = get_param(0);
	jit_value y = get_param(1);
	jit_value z = get_param(2);

	insn_return(x * y + z);
}

int main() {
	jit_int arg1,arg2,arg3;
	void *args[3];
	jit_int result;

	jit_context ctx;

	mul_add_function mul_add(ctx);

	arg1 = 3;
	arg2 = 5;
	arg3 = 2;
	args[0] = &arg1;
	args[1] = &arg2;
	args[2] = &arg3;
	mul_add.apply(args,&result);
	printf("mul_add(3,5,2) = %d\n",(int)result);

	arg1 = 13;
	arg2 = 5;
	arg3 = 7;
	args[0] = &arg1;
	args[1] = &arg2;
	args[2] = &arg3;
	mul_add.apply(args,&result);
	printf("mul_add(13,5,7) = %d\n",(int)result);

	mul_add.build_start();
	mul_add.build();
	mul_add.compile();
	mul_add.build_end();

	arg1 = 2;
	arg2 = 18;
	arg3 = -3;
	args[0] = &arg1;
	args[1] = &arg2;
	args[2] = &arg3;
	mul_add.apply(args,&result);
	printf("mul_add(2,18,-3) = %d\n",(int)result);
	
	return 0;
}
