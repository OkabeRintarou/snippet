#include <stdio.h>
#include <time.h>
#include <locale.h>
#include <sys/time.h>

#define BUF_SIZE  1000

int main()
{
	struct timeval tv;
	gettimeofday(&tv,NULL);
	printf("%ld:%ld\n",(long)tv.tv_sec,(long)tv.tv_usec);
	time_t t;
	time(&t);
	printf("%ld\n",(long)t);
	printf("%s\n",ctime(&t));

	struct tm *ptm;
	ptm = gmtime(&t);
	printf("%d %d %d\n",ptm->tm_year + 1900,ptm->tm_mon,ptm->tm_mday);
	ptm = localtime(&t);
	printf("%d %d %d\n",ptm->tm_year + 1900,ptm->tm_mon,ptm->tm_mday);
	time_t retTime = mktime(ptm);
	printf("%ld\n",retTime);
	printf("%s\n",asctime(ptm));
	
	char buf[BUF_SIZE];
	if(strftime(buf,BUF_SIZE,"%Y %m %e %T %P",ptm) != 0){
		printf("%s\n",buf);
	}
	printf("%s\n",setlocale(LC_ALL,""));
	return 0;
}
