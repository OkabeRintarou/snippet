#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

int main(int argc,char *argv[]) {
	DIR *dp;
	struct dirent *dirp;

	if (argc != 2) {
		fprintf(stderr,"usage: ls directory_name");
		exit(1);
	}
	if ((dp = opendir(argv[1])) == NULL) {
		fprintf(stderr,"can't open %s",argv[1]);
		exit(1);
	}
	while((dirp = readdir(dp)) != NULL) {
		printf("%s\n",dirp->d_name);
	}
	return 0;
}
