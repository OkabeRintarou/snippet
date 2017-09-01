#ifndef _PARSER_HPP
#define _PARSER_HPP

#include <string>
#include <memory>
#include <vector>


class ExprAST
{
public:
    virtual ~ExprAST() {}
};

class NumberExprAST: public ExprAST
{
private:
    double Val;
public:
    NumberExprAST(double val):Val(val) {}
};

class VariableExprAST: public ExprAST
{
private:
    std::string Name;
public:
    VariableExprAST(const std::string& name):Name(name) {}
};

class BinaryExprAST: public ExprAST
{
private:
    std::unique_ptr<ExprAST> LHS,RHS;
    char Op;
public:
    BinaryExprAST(char op,std::unique_ptr<ExprAST> lhs,std::unique_ptr<ExprAST> rhs)
            :LHS(std::move(lhs)),RHS(std::move(rhs)),Op(op)
    {

    }
};

class CallExprAST: public ExprAST
{
private:
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
public:
    CallExprAST(const std::string& callee,std::vector<std::unique_ptr<ExprAST>> args)
            :Callee(callee),Args(std::move(args))
    {

    }
};

class PrototypeAST
{
private:
    std::string Name;
    std::vector<std::string> Args;
public:
    PrototypeAST(const std::string& name,std::vector<std::string> args)
            :Name(name),Args(std::move(args))
    {

    }

    const std::string& getName()const {return Name;}
};

class FunctionAST
{
private:
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;
public:
    FunctionAST(std::unique_ptr<PrototypeAST> proto,std::unique_ptr<ExprAST> body)
            :Proto(std::move(proto)),Body(std::move(body))
    {

    }
};

extern int CurTok;
extern int getNextToken();
extern std::unique_ptr<ExprAST> ParseNumberExpr();
extern std::unique_ptr<ExprAST> ParseParenExpr();
extern std::unique_ptr<ExprAST> ParseIdentifierExpr();
extern std::unique_ptr<ExprAST> ParsePrimary();
extern std::unique_ptr<ExprAST> ParseExpression();
extern std::unique_ptr<ExprAST> ParseExpression();
extern std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,std::unique_ptr<ExprAST> LHS);
extern std::unique_ptr<PrototypeAST> ParsePrototype();
extern std::unique_ptr<FunctionAST> ParseDefinition();
extern std::unique_ptr<PrototypeAST> ParseExtern();
extern std::unique_ptr<FunctionAST> ParseTopLevelExpr();


#endif