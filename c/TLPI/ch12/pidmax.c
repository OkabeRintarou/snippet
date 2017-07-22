#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>

#define MAXLINE 4096

int main(int argc,char* argv[])
{
	int fd;
	char line[MAXLINE];
	int n;

	fd = open("/proc/sys/kernel/pid_max",(argc > 1) ? O_RDWR : O_RDONLY);
	if(fd < 0) {
		fprintf(stderr,"open:%s\n",strerror(errno));
		return -1;
	}
	
	n = read(fd,line,MAXLINE);
	if(n < 0){
		fprintf(stderr,"open:%s\n",strerror(errno));
		return -1;
	}	

	if(argc > 1){
		printf("Old value: ");
	}
	printf("%.*s",(int)n,line);
	if(argc > 1){
		if(write(fd,argv[1],strlen(argv[1])) != strlen(argv[1])){
			fprintf(stderr,"open:%s\n",strerror(errno));
			return -1;
		}
		system("echo /proc/sys/kernel/pid_max now contains "
				"`cat /proc/sys/kernel/pid_max`");
	}
	return 0;
}
