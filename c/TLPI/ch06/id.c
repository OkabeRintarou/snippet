#include <unistd.h>
#include <stdio.h>

int main()
{
	printf("getpid(): %ld\n",(long)getpid());
	printf("getppid(): %ld\n",(long)getppid());
	return 0;
}
