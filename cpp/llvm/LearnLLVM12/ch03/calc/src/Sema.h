#pragma once

#include "AST.h"
#include "Parser.h"

class Sema {
public:
    bool semantic(AST *tree);
};
