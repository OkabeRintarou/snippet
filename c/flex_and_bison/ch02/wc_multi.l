%option noyywrap
%{
#include <stdio.h>
#include <string.h>

int chars = 0,lines = 0,words = 0;
int total_chars = 0,total_lines = 0,total_words = 0;
%}

%%

[a-zA-Z]+   { ++words; chars += strlen(yytext); }
\n					{ ++lines; ++chars; }
.						{ ++chars; }

%%

void print_result(const char *filename) {
	if (!filename) {
		printf("# of lines %8d\n# of words %8d\n# of chars %8d\n",lines,words,chars);
	} else {
		printf("%s:\n# of lines %8d\n# of words %8d\n# of chars %8d\n",filename,lines,words,chars);
	}
}

int main(int argc,char *argv[]) {
	int i;
	if (argc < 2) {
		yylex();
		print_result(NULL);
	}

	for (i = 1; i < argc; i++) {
		FILE *f = fopen(argv[i],"r");
		if (!f) {
			perror("fopen error");
			return 1;
		}

		yyrestart(f);
		yylex();
		fclose(f);
		print_result(argv[i]);
		total_lines += lines;
		total_words += words;
		total_chars += chars;
		lines = words = chars = 0;
	}
	if (argc > 1) {
		printf("# total of lines %8d\n# total of words %8d\n# total of chars %8d\n",
					total_lines,total_words,total_chars);
	}
	return 0;
}
