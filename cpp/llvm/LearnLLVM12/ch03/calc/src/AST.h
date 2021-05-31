#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

class AST;
class Expr;
class Factor;
class BinaryOp;
class WithDecl;

class ASTVisitor {
public:
    virtual void visit(AST &) = 0;
    virtual void visit(Expr &) = 0;
    virtual void visit(Factor &) = 0;
    virtual void visit(BinaryOp &) = 0;
    virtual void visit(WithDecl &) = 0;
};

class AST {
public:
    virtual ~AST() = default;
    virtual void accept(ASTVisitor &V) = 0;
};

class Expr : public AST {
public:
    Expr() = default;
};

class Factor : public Expr {
public:
    enum ValueKind { Ident, Number };
private:
    ValueKind Kind;
    llvm::StringRef Value;
public:
    Factor(ValueKind kind, llvm::StringRef val) : Kind(kind), Value(val) {}

    ValueKind getKind() const { return Kind; }
    llvm::StringRef getVal() const { return Value; }

    virtual void accept(ASTVisitor &V) override {
        V.visit(*this);
    }
};

class BinaryOp : public Expr {
public:
    enum Operator { Plus, Minus, Mul, Div };
private:
    Expr *Lhs;
    Expr *Rhs;
    Operator Op;
public:
    BinaryOp(Operator op, Expr *l, Expr *r) : Lhs(l), Rhs(r), Op(op) {}
    Expr *getLeft() const { return Lhs; }
    Expr *getRight() const { return Rhs; }
    Operator getOperator() const { return Op; }

    virtual void accept(ASTVisitor &V) override {
        V.visit(*this);
    }
};

class WithDecl : public AST {
    using VarVector = llvm::SmallVector<llvm::StringRef, 8>;
    VarVector  Vars;
    Expr *E;
public:
    WithDecl(llvm::SmallVector<llvm::StringRef, 8> Vars, Expr *E) : Vars(Vars), E(E) {}
    VarVector::const_iterator begin() const { return Vars.begin(); }
    VarVector ::const_iterator end() const { return Vars.end(); }
    Expr *getExpr() const { return E; }

    virtual void accept(ASTVisitor &V) override {
        V.visit(*this);
    }
};