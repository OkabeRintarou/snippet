#include <stdio.h>
#include <stdlib.h>
#include <pwd.h>
#include <unistd.h>
#include <errno.h>
#include <shadow.h>
#include <limits.h>
#include <malloc.h>
#include <string.h>
#include <crypt.h>

int main()
{
	char *username,*password,*encrypted,*p;
	struct passwd *pwd;
	struct spwd *spwd;
	size_t len;
	long lnmax;
	lnmax = sysconf(_SC_LOGIN_NAME_MAX);
	if(lnmax == -1){
		lnmax = 256;
	}
	username = (char*)malloc(lnmax);
	if(username == NULL){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}
	printf("Username: ");
	fflush(stdout);
	if(fgets(username,lnmax,stdin) == NULL){
		exit(EXIT_FAILURE);
	}
	len = strlen(username);
	if(username[len - 1] == '\n')
		username[len - 1] = '\0';

	pwd = getpwnam(username);
	if(pwd == NULL){
		fprintf(stderr,"couldn't get password record");
		exit(EXIT_FAILURE);
	}
	spwd = getspnam(username);
	if(spwd == NULL && errno == EACCES){
		fprintf(stderr,"no permission to read shadown password file");
		exit(EXIT_FAILURE);
	}
	if(spwd != NULL){
		pwd->pw_passwd = spwd->sp_pwdp;
	}
	password = getpass("Password: ");
	encrypted = crypt(password,pwd->pw_passwd);
	for(p = password;*p != '\0';){
		*p++ = '\0';
	}

	if(encrypted == NULL){
		fprintf(stderr,"crypt error\n");
		exit(EXIT_FAILURE);
	}
	if(strcmp(encrypted,pwd->pw_passwd) == 0){
		printf("Successfully authenticated: UID=%ld\n",(long)pwd->pw_uid);
	}else{
		printf("Incorrect password\n");
		exit(EXIT_FAILURE);
	}

	return 0;
}
