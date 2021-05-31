#include "Lexer.h"

namespace charinfo {

LLVM_READNONE inline bool isWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\f' || c == '\v' ||
           c == '\r' || c == '\n';
}

LLVM_READNONE inline bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

LLVM_READNONE inline bool isLetter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

}

void Lexer::next(Token &token) {
    char c;
    while ((c = *BufferPtr) && charinfo::isWhitespace(c)) {
        ++BufferPtr;
    }

    if (!c) {
        token.Kind = Token::eoi;
        return;
    }

    if (charinfo::isLetter(c)) {
        const char *end = BufferPtr + 1;
        while (charinfo::isLetter(*end)) {
            ++end;
        }
        llvm::StringRef name(BufferPtr, end - BufferPtr);
        Token::TokenKind kind =
                name == "with" ? Token::KW_with : Token::ident;
        formToken(token, end, kind);
        return;
    } else if (charinfo::isDigit(c)) {
        const char *end = BufferPtr + 1;
        while (charinfo::isDigit(*end)) {
            ++end;
        }
        formToken(token, end, Token::number);
        return;
    } else {
        switch (c) {
#define CASE(ch, tok) \
case ch: formToken(token, BufferPtr + 1, tok); break
            CASE('+', Token::plus);
            CASE('-', Token::minus);
            CASE('*', Token::star);
            CASE('/', Token::slash);
            CASE('(', Token::l_paren);
            CASE(')', Token::r_paren);
            CASE(':', Token::colon);
            CASE(',', Token::comma);
#undef CASE
            default:
                formToken(token, BufferPtr + 1, Token::unknow);
        }
    }

}

void Lexer::formToken(Token &token, const char *TokEnd, Token::TokenKind Kind) {
    token.Kind = Kind;
    token.Text = llvm::StringRef(BufferPtr, TokEnd - BufferPtr);
    BufferPtr = TokEnd;
}