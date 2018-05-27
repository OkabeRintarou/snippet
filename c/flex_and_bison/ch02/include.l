%option noyywrap
%x IFILE

%{
	#include <malloc.h>
	#include <stdio.h>
	#include <stdlib.h>

	struct bufstack {
		struct bufstack *prev;
		YY_BUFFER_STATE bs;
		int lineno;
		char *filename;
		FILE *f;
	}*curbs = NULL;

	const char *curfilename;

	int newfile(const char *fn);
	int popfile(void);
%}

%%
^"#"[ \t]*include[ \t]*[\"<]			{ BEGIN IFILE; }

<IFILE>[^ \t\n\">]+								{
																			{
																				int c;
																				while ((c = input()) != '\n');
																			}
																			yylineno++;
																			if (!newfile(yytext)) {
																				yyterminate(); /* no such file */
																			}
																			BEGIN INITIAL;
																	}

<IFILE>.|\n                       {
																			fprintf(stderr,"%4d bad include line\n",yylineno);
																			yyterminate();
																	}

<<EOF>>														{ if (!popfile()) yyterminate(); }

^.                                { fprintf(yyout,"%4d %s",yylineno,yytext); }
^\n																{ fprintf(yyout,"%4d %s",yylineno++,yytext); }
\n																{ ECHO; yylineno++; }
.																	{ ECHO; }

%%
int main(int argc,char *argv[]) {
	if (argc < 2) {
		fprintf(stderr,"need filename\n");
		return 1;
	}
	if (newfile(argv[1])) {
		yylex();
	}
	return 0;
}

int newfile(const char *fn) {
	char filename[1024];
	FILE *f = fopen(fn,"r");
	struct bufstack *bs = (struct bufstack*)malloc(sizeof(struct bufstack));
	if (!f) {
		sprintf(filename,"/usr/include/%s",fn);
		if ((f = fopen(filename,"r")) == NULL) {
			sprintf(filename,"fopen %s error",fn);
			perror(filename);
			return 0;
		}
	}
	if (!bs) {
		perror("malloc error");
		exit(1);
	}
	if (curbs) curbs->lineno = yylineno;
	bs->prev = curbs;

	bs->bs = yy_create_buffer(f,YY_BUF_SIZE);
	bs->f = f;
	yy_switch_to_buffer(bs->bs);
	curbs = bs;
	yylineno = 1;
	curfilename = fn;
	return 1;
}

int popfile(void) {
	struct bufstack *bs = curbs;
	struct bufstack *prevbs;

	if (!bs) return 0;

	/* close current file */
	fclose(bs->f);
	yy_delete_buffer(bs->bs);

	/* switch back to previous */
	prevbs = bs->prev;
	free(bs);

	if (!prevbs) return 0;

	yy_switch_to_buffer(prevbs->bs);
	curbs = prevbs;
	yylineno = curbs->lineno;
	curfilename = curbs->filename;
	return 1;
}
