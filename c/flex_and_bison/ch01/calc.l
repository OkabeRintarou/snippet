%{
#include "calc.tab.h"
%}

%%
"+"		{ return ADD; }
"-"		{ return SUB; }
"*"   { return MUL; }
"/"		{ return DIV; }
"|"		{ return ABS; }
"("		{ return OP; }
")"		{ return CP; }
"//".* { }
[0-9]+	{ yylval = atoi(yytext); return NUMBER; }
0[xX][a-f0-9]+   { yylval = strtol(yytext,NULL,16); return NUMBER; }
\n			{ return EOL; }
.				{ printf("Mystery character %c\n", *yytext); }

%%

