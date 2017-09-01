#ifndef _LEXER_HPP
#define _LEXER_HPP

#include <string>

enum Token {
    tok_eof = -1,

    tok_def = -2,
    tok_extern = -3,

    tok_identifier = -4,
    tok_number = -5,
};

extern std::string IdentifierStr;
extern double NumVal;

extern int gettok();

#endif
