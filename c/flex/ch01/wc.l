%{
#include <stdio.h>
#include <string.h>

int chars = 0;
int words = 0;
int lines = 0;

%}

%%

[a-zA-z]+   {
							words++;
							chars = strlen(yytext);
						}
\n          {
							chars++;
							lines++;
						}
.       		{ chars++; }

%%

int main() {
	yylex();
	printf("# of lines %8d\n# of words %8d\n# of chars %8d\n",
		lines,words,chars);
	return 0;
}

