#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

class Lexer;

class Token {
    friend class Lexer;

public:
    enum TokenKind : unsigned short {
        eoi,
        unknow,
        ident,
        number,
        comma,
        colon,
        plus,
        minus,
        star,
        slash,
        l_paren,
        r_paren,
        KW_with,
    };
private:
    TokenKind Kind;
    llvm::StringRef Text;
public:
    TokenKind getKind() const { return Kind; }

    llvm::StringRef getText() const { return Text; }

    bool is(TokenKind kind) const { return Kind == kind; }

    bool isOneOf(TokenKind k1, TokenKind k2) const {
        return Kind == k1 || Kind == k2;
    }

    template<typename... Ts>
    bool isOneOf(TokenKind k1, TokenKind k2, Ts... Ks) const {
        return is(k1) || isOneOf(k2, Ks...);
    }
};

class Lexer {
    const char *BufferStart;
    const char *BufferPtr;
public:
    Lexer(const llvm::StringRef &Buffer) {
        BufferStart = Buffer.begin();
        BufferPtr = BufferStart;
    }

    void next(Token &token);

private:
    void formToken(Token &token, const char *TokEnd, Token::TokenKind Kind);
};