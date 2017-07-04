#include <assert.h>
#include <stdio.h>

#define _BSD_SOURCE
#include <stdlib.h>

extern char **environ;

int main(int argc,char *argv[])
{
	char **ep;
	for(ep = environ;*ep != NULL;ep++){
		puts(*ep);
	}

	const char *shell = getenv("SHELL");
	printf("\n\n%s\n",shell == NULL ? "<null>" : shell);
	if(putenv("SHELL=/usr/bin/shell") < 0){
		fprintf(stderr,"putenv failed\n");
	}else{
		shell = getenv("SHELL");
		assert(shell != NULL);
		printf("SHELL=%s\n",shell);
	}
	unsetenv("SHELL");
	shell = getenv("SHELL");
	assert(shell == NULL);
	return 0;
}
