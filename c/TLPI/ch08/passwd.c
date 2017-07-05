#include <stdio.h>
#include <errno.h>
#include <pwd.h>

int main()
{
	struct passwd *pwd;
	pwd = getpwnam("syl");
	if(pwd == NULL){
		if(errno == 0){
			fprintf(stderr,"Could not find user named `syl`\n");
		}else{
			fprintf(stderr,"getpwnam error\n");
		}
	}else{
		printf("name:%s\n",pwd->pw_name);
		printf("passwd:%s\n",pwd->pw_passwd);
		printf("uid:%d\n",(int)pwd->pw_uid);
		printf("gid:%d\n",(int)pwd->pw_gid);
		printf("comment:%s\n",pwd->pw_gecos);
		printf("dir:%s\n",pwd->pw_dir);
		printf("shell:%s\n",pwd->pw_shell);
		printf("\n\n");
	}
	pwd = getpwuid(1000);
	if(pwd == NULL){
		if(errno == 0){
			fprintf(stderr,"Could not find user named `syl`\n");
		}else{
			fprintf(stderr,"getpwnam error\n");
		}
	}else{
		printf("name:%s\n",pwd->pw_name);
		printf("passwd:%s\n",pwd->pw_passwd);
		printf("uid:%d\n",(int)pwd->pw_uid);
		printf("gid:%d\n",(int)pwd->pw_gid);
		printf("comment:%s\n",pwd->pw_gecos);
		printf("dir:%s\n",pwd->pw_dir);
		printf("shell:%s\n",pwd->pw_shell);
		printf("\n\n");
	}
	return 0;
}

