#include <errno.h>
#include <stdio.h>
#include <grp.h>

int main()
{
	struct group *grp;
	grp = getgrnam("syl");
	if(grp == NULL){
		if(errno == 0){
			fprintf(stderr,"Could not find group named `syl`\n");
		}else{
			fprintf(stderr,"getgrname error\n");
		}
	}else{
		printf("name: %s\n",grp->gr_name);
		printf("passwd: %s\n",grp->gr_passwd);
		printf("gid: %d\n",(int)grp->gr_gid);
		printf("member list: \n");
		char **mem = grp->gr_mem;
		while(*mem){
			printf("\t%s\n",*mem);
			++mem;
		}
	}
	return 0;
}
