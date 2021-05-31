#include "AST.h"
#include "Sema.h"
#include "llvm/ADT/StringSet.h"

namespace {

class DeclCheck : public ASTVisitor {
    llvm::StringSet<> Scope;
    bool HasError;

    enum ErrorType {
        Twice, Not
    };

    void error(ErrorType ET, llvm::StringRef V) {
        llvm::errs() << "Variable " << V << " "
                     << (ET == Twice ? "already" : "not")
                     << " declared\n";

    }

public:
    DeclCheck() : HasError(false) {}

    bool hasError() const { return HasError; }

    virtual void visit(Factor &Node) override {
        if (Node.getKind() == Factor::Ident) {
            if (Scope.find(Node.getVal()) == Scope.end()) {
                error(Not, Node.getVal());
            }
        }
    }

    virtual void visit(BinaryOp &Node) override {
        if (Node.getLeft()) {
            Node.getLeft()->accept(*this);
        } else {
            HasError = true;
        }
        if (Node.getRight()) {
            Node.getRight()->accept(*this);
        } else {
            HasError = true;
        }
    }

    virtual void visit(WithDecl &Node) override {
        for (auto I = Node.begin(), E = Node.end(); I != E; ++I) {
            if (!Scope.insert(*I).second) {
                error(Twice, *I);
            }
        }
        if (Node.getExpr()) {
            Node.getExpr()->accept(*this);
        } else {
            HasError = true;
        }
    }

    virtual void visit(AST &) {}
    virtual void visit(Expr &) {}
};

} // end namespace

bool Sema::semantic(AST *tree) {
    if (tree == nullptr) {
        return false;
    }
    DeclCheck check;
    tree->accept(check);
    return check.hasError();
}
