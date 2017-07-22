#define _GNU_SOURCE
#include <sys/utsname.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int main()
{
	struct utsname uts;
	if(uname(&uts) < 0){
		fprintf(stderr,"uname:%s\n",strerror(errno));
		exit(EXIT_FAILURE);
	}
	printf("Node name:		%s\n",uts.nodename);
	printf("System name:	%s\n",uts.sysname);
	printf("Release:		%s\n",uts.release);
	printf("Version:		%s\n",uts.version);
	printf("machine:		%s\n",uts.machine);
#ifdef _GNU_SOURCE
	printf("domainname:		%s\n",uts.domainname);
#endif
	return 0;
}
