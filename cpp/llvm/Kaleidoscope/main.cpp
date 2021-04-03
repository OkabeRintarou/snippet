#include <cstdio>
#include <llvm/IR/Function.h>
#include "lexer.h"
#include "parser.h"
#include "codegen.h"

using namespace llvm;

static void HandleDefinition() {
    if (auto FnAST = ParseDefinition()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "Read function definition:");
            FnIR->print(errs());
            fprintf(stderr, "\n");
        }
    } else {
        getNextToken();
    }
}

static void HandleExtern() {
    if (auto ProtoAST = ParseExtern()) {
        if (auto *FnIR = ProtoAST->codegen()) {
            fprintf(stderr, "Read extern: ");
            FnIR->print(errs());
            fprintf(stderr, "\n");
        }

    } else {
        getNextToken();
    }
}

static void HandleTopLevelExpression() {
    if (auto FnAST = ParseTopLevelExpr()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "Read top-level expression:");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            FnIR->eraseFromParent();
        }
    } else {
        getNextToken();
    }
}

static void MainLoop() {
    for (;;) {
        fprintf(stderr, "ready>");
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

int main(int argc, char *argv[]) {
    // freopen("test/fib.kal","r",stdin);
    fprintf(stderr, "read>");
    getNextToken();

    BeginCodegen();
    MainLoop();
    EndCodegen();

    return 0;
}