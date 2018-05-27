%option noyywrap
%{
#include <stdio.h>
#include <string.h>

int chars = 0,lines = 0,words = 0;
%}

%%

[a-zA-Z]+   { ++words; chars += strlen(yytext); }
\n					{ ++lines; ++chars; }
.						{ ++chars; }

%%
int main(int argc,char *argv[]) {
	if (argc > 1) {
		if((yyin = fopen(argv[1],"r")) == NULL) {
			perror(argv[1]);
			return 1;
		}
	}

	yylex();
	printf("# of lines %8d\n# of words %8d\n# of chars %8d\n",lines,words,chars);
	return 0;
}
