#include <stdio.h>
#include <setjmp.h>

static jmp_buf env;

static void x()
{
	switch(setjmp(env)){
	case 0:
		printf("Call from x()\n");
		break;
	case 1:
		printf("Return from longjmp\n");
		break;
	}
	printf("Return from x()\n");
}

static void y()
{
	longjmp(env,1);
}

int main()
{
	x();
	y();
	return 0;
}
