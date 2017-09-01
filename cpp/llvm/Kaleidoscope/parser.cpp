#include <cstdio>
#include <memory>
#include <unordered_map>

#include <llvm/ADT/STLExtras.h>


#include "lexer.h"
#include "parser.h"

int CurTok;
static std::unordered_map<char,int> BinopPrecedence = {
        {'<',10},
        {'+',20},
        {'-',20},
        {'*',40},
};

int getNextToken()
{
    return CurTok = gettok();
}

static int getTokPrecedence()
{
    if(!isascii(CurTok)) {
        return -1;
    }

    int TokPrec = BinopPrecedence[CurTok];
    if(TokPrec <= 0) {
        return -1;
    }
    return TokPrec;
}

std::unique_ptr<ExprAST> LogError(const char* Str)
{
    fprintf(stderr,"LogError: %s\n",Str);
    return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char* Str)
{
    LogError(Str);
    return nullptr;
}

/// numberexpr ::= number
std::unique_ptr<ExprAST> ParseNumberExpr()
{
    auto Result = llvm::make_unique<NumberExprAST>(NumVal);
    getNextToken();
    return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
std::unique_ptr<ExprAST> ParseParenExpr()
{
    getNextToken(); // eat '('
    auto V = ParseExpression();
    if(!V) {
        return nullptr;
    }
    if(CurTok != ')') {
        return LogError("expected ')");
    }
    getNextToken(); // eat ')'
    return V;
}

/// identifierexpr
///     ::= identifier
///     ::= identifier '(' expression* ')'
std::unique_ptr<ExprAST> ParseIdentifierExpr()
{
    std::string IdName = IdentifierStr;

    getNextToken(); // eat identifier

    if(CurTok != '(') {
        return llvm::make_unique<VariableExprAST>(IdName);
    }

    // Call
    getNextToken(); // eat '('
    std::vector<std::unique_ptr<ExprAST>> Args;
    if(CurTok != ')') {
        while(true) {
            if(auto Arg = ParseExpression()) {
                Args.push_back(std::move(Arg));
            } else {
                return nullptr;
            }
            if(CurTok == ')') {
                break;
            }
            if(CurTok != ',') {
                return LogError("Expected ')' or ',' in argument list");
            }
            getNextToken();
        }
    }

    // Eat the ')'
    getNextToken();

    return llvm::make_unique<CallExprAST>(IdName,std::move(Args));
}

/// primary
///     ::= identifierexpr
///     ::= numberexpr
///     ::= parenexpr
std::unique_ptr<ExprAST> ParsePrimary()
{
    switch (CurTok) {
        case tok_identifier:
            return ParseIdentifierExpr();
        case tok_number:
            return ParseNumberExpr();
        case '(':
            return ParseParenExpr();
        default:
            return LogError("unknow token when expecting an expression");
    }
}


/// expression
///     ::= primary binoprhs
std::unique_ptr<ExprAST> ParseExpression()
{
    auto LHS = ParsePrimary();
    if(!LHS) {
        return nullptr;
    }
    return ParseBinOpRHS(0,std::move(LHS));
}

/// binoprhs
///     ::= ('+',primary)*
std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,std::unique_ptr<ExprAST> LHS)
{
    for(;;) {
        int TokPrec = getTokPrecedence();

        if(TokPrec < ExprPrec) {
            return LHS;
        }

        int BinOp = CurTok;
        getNextToken();

        auto RHS = ParsePrimary();
        if(!RHS) {
            return nullptr;
        }

        int NextPrec = getTokPrecedence();
        if(TokPrec < NextPrec) {
            RHS = ParseBinOpRHS(TokPrec + 1,std::move(RHS));
            if(!RHS) {
                return nullptr;
            }
        }

        LHS = llvm::make_unique<BinaryExprAST>(BinOp,std::move(LHS),std::move(RHS));
    }
}

/// prototype
///     ::= id '(' id* ')'
std::unique_ptr<PrototypeAST> ParsePrototype()
{
    if(CurTok != tok_identifier) {
        return LogErrorP("Expected function name in prototype");
    }
    std::string FnName = IdentifierStr;
    getNextToken();

    if(CurTok != '(') {
        return LogErrorP("Expected '(' in prototype");
    }

    std::vector<std::string> ArgNames;
    while(getNextToken() == tok_identifier) {
        ArgNames.push_back(IdentifierStr);
    }
    if(CurTok != ')') {
        return LogErrorP("Expected ')' in prototype");
    }

    // eat ')'
    getNextToken();
    return llvm::make_unique<PrototypeAST>(FnName,std::move(ArgNames));
}

/// definition ::= 'def' prototype expression
std::unique_ptr<FunctionAST> ParseDefinition()
{
    getNextToken(); // eat 'def'
    auto Proto = ParsePrototype();
    if(!Proto) {
        return nullptr;
    }
    if(auto E = ParseExpression()) {
        return llvm::make_unique<FunctionAST>(std::move(Proto),std::move(E));
    }
    return nullptr;
}

/// external ::= 'extern' prototype
std::unique_ptr<PrototypeAST> ParseExtern()
{
    getNextToken();
    return ParsePrototype();
}

/// toplevelexpr ::= expression
std::unique_ptr<FunctionAST> ParseTopLevelExpr()
{
    if(auto E = ParseExpression()) {
        auto Proto = llvm::make_unique<PrototypeAST>("",std::vector<std::string>());
        return llvm::make_unique<FunctionAST>(std::move(Proto),std::move(E));
    }
    return nullptr;
}
