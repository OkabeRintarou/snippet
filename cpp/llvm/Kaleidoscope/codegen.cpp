#include <string>
#include <map>
#include <memory>
#include <vector>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>

#include "error.h"
#include "parser.h"
#include "codegen.h"

using namespace llvm;

static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<IRBuilder<>> Builder;
static std::unique_ptr<Module> TheModule;
static std::map<std::string, Value *> NamedValues;

Value *NumberExprAST::codegen() {
    return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
    Value *V = NamedValues[Name];
    if (!V) {
        LogErrorV("Unknown variable name");
    }
    return V;
}

Value *BinaryExprAST::codegen() {
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();

    if (!L || !R) {
        return nullptr;
    }

    switch (Op) {
        case '+':
            return Builder->CreateFAdd(L, R, "addtmp");
        case '-':
            return Builder->CreateFSub(L, R, "subtmp");
        case '*':
            return Builder->CreateFMul(L, R, "multmp");
        case '<':
            L = Builder->CreateFCmpULT(L, R, "cmptmp");
            // Convert bool 0/1 to double 0.0 or 1.0
            return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext),
                                        "booltmp");
        default:
            return LogErrorV("invalid binary operator");
    }
}

Value *CallExprAST::codegen() {
    Function *CalleeF = TheModule->getFunction(Callee);
    if (!CalleeF) {
        return LogErrorV("Unknown function reference");
    }

    // If argument mismatch error
    if (CalleeF->arg_size() != Args.size()) {
        return LogErrorV("Incorrect # arguments passed");
    }

    std::vector<Value *> ArgsV;
    for (size_t i = 0, e = Args.size(); i != e; i++) {
        ArgsV.emplace_back(Args[i]->codegen());
        if (!ArgsV.back()) {
            return nullptr;
        }
    }

    return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Function *PrototypeAST::codegen() {
    std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
    FunctionType *FT =
            FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);
    Function *F =
            Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());
    size_t Idx = 0;
    for (auto &Arg : F->args()) {
        Arg.setName(Args[Idx++]);
    }
    return F;
}

Function *FunctionAST::codegen() {
    // First, check for an existing function from a previous 'extern' declaration
    Function *TheFunction = TheModule->getFunction(Proto->getName());

    if (!TheFunction) {
        TheFunction = Proto->codegen();
    }
    if (!TheFunction) {
        return nullptr;
    }

    if (!TheFunction->empty()) {
        return (Function *) LogErrorV("Function cannot be redefined");
    }
    BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    NamedValues.clear();
    for (auto &Arg : TheFunction->args()) {
        NamedValues[Arg.getName().str()] = &Arg;
    }

    if (Value *retValue = Body->codegen()) {
        Builder->CreateRet(retValue);
        verifyFunction(*TheFunction);
        return TheFunction;
    }

    // Error reading body, remove function
    TheFunction->eraseFromParent();
    return nullptr;
}

void BeginCodegen() {
    TheContext = std::make_unique<LLVMContext>();
    TheModule = std::make_unique<Module>(StringRef("my cool jit"), *TheContext);

    Builder = std::make_unique<IRBuilder<>>(*TheContext);
}

void EndCodegen() {
    TheModule->print(errs(), nullptr);
}