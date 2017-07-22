#include <stdio.h>
#include <limits.h>

struct limit_arg
{
	char* name;
	int number;
};

struct limit_arg limit_args[] = {
//	{"ARG_MAX",ARG_MAX},
	{"LOGIN_NAME_MAX",LOGIN_NAME_MAX},
//	{"OPEN_MAX",OPEN_MAX},
	{"NGROUPS_MAX",NGROUPS_MAX},
	{"RTSIG_MAX",RTSIG_MAX},
//	{"SIGQUEUE_MAX",SIGQUEUE_MAX},
//	{"STREAM_MAX",STREAM_MAX},
	{"NAME_MAX",NAME_MAX},
	{"PATH_MAX",PATH_MAX},
	{"PIPE_BUF",PIPE_BUF},
};

int main()
{
	for(int i = 0;i < sizeof(limit_args) / sizeof(*limit_args);i++){
		printf("%s:%d\n",limit_args[i].name,limit_args[i].number);
	}
	return 0;
}
