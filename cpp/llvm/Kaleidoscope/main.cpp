#include <cstdio>
#include "lexer.h"
#include "parser.h"

static void HandleDefinition()
{
    if(ParseDefinition()) {
        fprintf(stderr,"Parsed a function definition.\n");
    } else {
        getNextToken();
    }
}

static void HandleExtern()
{
    if(ParseExtern()) {
        fprintf(stderr,"Parsed an extern,\n");
    } else {
        getNextToken();
    }
}

static void HandleTopLevelExpression()
{
    if(ParseTopLevelExpr()) {
        fprintf(stderr,"Parsed a top-level expr.\n");
    } else {
        getNextToken();
    }
}

static void MainLoop()
{
    for(;;) {
        fprintf(stderr,"ready>");
        switch (CurTok) {
            case tok_eof:
                return;
            case ';':
                getNextToken();
                break;
            case tok_def:
                HandleDefinition();
                break;
            case tok_extern:
                HandleExtern();
                break;
            default:
                HandleTopLevelExpression();
                break;
        }
    }
}

int main(int argc,char* argv[])
{
    // freopen("test/fib.kal","r",stdin);
    fprintf(stderr,"read>");
    getNextToken();
    MainLoop();

    return 0;
}