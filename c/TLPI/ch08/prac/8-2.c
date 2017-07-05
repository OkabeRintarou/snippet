#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <pwd.h>
#include <string.h>

struct passwd* mygetpwnam(const char *name)
{
	struct passwd *pwd;
	while((pwd = getpwent()) != NULL){
		if(strcmp(pwd->pw_name,name) == 0){
			endpwent();
			return pwd;
		}
	}
	endpwent();
	return NULL;
}

int main()
{
	struct passwd *cpwd = getpwnam("syl");
	struct passwd *mypwd = mygetpwnam("syl");
	assert((cpwd != NULL && mypwd != NULL) || (cpwd == NULL && mypwd == NULL));
	if(cpwd != NULL){
		assert(strcmp(cpwd->pw_name,mypwd->pw_name) == 0);
		assert(cpwd->pw_uid == mypwd->pw_uid);
	}
}
