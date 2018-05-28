#ifndef CALC_H
#define CALC_H

extern int yylineno;
void yyerror(char *s,...);

struct symbol {
	char *name;
	double value;
	struct ast *func;
	struct symlist *syms;
};

#define NHASH 9997

struct symbol symtab[NHASH];

struct symbol *lookup(const char *);

/* list of symbols, for an argument list */
struct symlist {
	struct symbol *sym;
	struct symlist *next;
};

struct symlist *newsymlist(struct symbol *sym,struct symlist *next);
void symlistfree(struct symlist *sl);

/* nodes in the abstract syntax tree */

/* node types
 * + - * / | 
 * M unary minus
 * L expression or statement list
 * I IF statement
 * W WHILE statement
 * N symbol ref
 * = assignment
 * S list of symbols
 * F built in function call
 * C user function call
 */

enum bifs {
	B_sqrt = 1,
	B_exp,
	B_log,
	B_print,
};


struct ast {
	int nodetype;
	struct ast *l;
	struct ast *r;
};

struct fncall {
	int nodetype;
	struct ast *l;
	enum bifs functype;
};

struct ufncall {
	int nodetype;
	struct ast *l;
	struct symbol *s;
};

struct flow {
	int nodetype;     /* type I or W */
	struct ast *cond; /* condition */
	struct ast *tl;   /* then branch or do list */
	struct ast *el;   /* optional else branch */
};

struct numval {
	int nodetype;     /* type K */
	double number;
};

struct symref {
	int nodetype;     /* type N */
	struct symbol *s;
};

struct symasgn {    /* type = */
	int nodetype;
	struct symbol *s;
	struct ast *v;
};

/* build an AST */
struct ast *newast(int nodetype,struct ast *l,struct ast *r);
struct ast *newcmp(int cmptype,struct ast *l,struct ast *r);
struct ast *newfunc(int functype,struct ast *l);
struct ast *newcall(struct symbol *s,struct ast *l);
struct ast *newref(struct symbol *s);
struct ast *newasgn(struct symbol *s,struct ast *v);
struct ast *newnum(double d);
struct ast *newflow(int nodetype,struct ast *cond,struct ast *tl,struct ast *tr);

void dodef(struct symbol *sym,struct symlist *syms,struct ast *stmts);

double eval(struct ast *);

void treefree(struct ast *);

#endif
