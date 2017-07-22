#include <stdio.h>
#include <unistd.h>

struct conf_arg
{
	char* name;
	int arg;
};

struct conf_arg conf_args[] = 
{
	{"_SC_ARG_MAX",_SC_ARG_MAX},
	{"_SC_LOGIN_NAME_MAX",_SC_LOGIN_NAME_MAX},
	{"_SC_OPEN_MAX",_SC_OPEN_MAX},
	{"_SC_PAGE_SIZE",_SC_PAGESIZE},
};

int main() 
{
	for(int i = 0;i < sizeof(conf_args) / sizeof(*conf_args);i++){
		printf("%20s:%ld\n",conf_args[i].name,sysconf(conf_args[i].arg));
	}
	return 0;
}
