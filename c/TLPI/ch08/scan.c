#include <stdio.h>
#include <pwd.h>

int main()
{	
	struct passwd *pwd;
	while((pwd = getpwent()) != NULL) {
		printf("%-25s %5ld\n",pwd->pw_name,(long)pwd->pw_uid);
	}
	endpwent();
	return 0;
}
