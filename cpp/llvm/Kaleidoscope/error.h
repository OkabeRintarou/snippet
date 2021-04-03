#pragma once

#include <memory>
#include <llvm/IR/Value.h>
#include "parser.h"

inline std::unique_ptr<ExprAST> LogError(const char *Str) {
    fprintf(stderr, "LogError: %s\n", Str);
    return nullptr;
}

inline std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
    LogError(Str);
    return nullptr;
}

inline llvm::Value *LogErrorV(const char *Str) {
    LogError(Str);
    return nullptr;
}