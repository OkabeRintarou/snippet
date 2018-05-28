#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"

struct ast *newast(int nodetype,struct ast *l,struct ast *r) {
	struct ast *n = (struct ast*)malloc(sizeof(struct ast));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = nodetype;
	n->l = l;
	n->r = r;
	return n;
}

struct ast *newcmp(int cmptype,struct ast *l,struct ast *r) {
	struct ast *n = (struct ast*)malloc(sizeof(struct ast));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = '0' + cmptype;
	n->l = l;
	n->r = r;
	return n;
}

struct ast *newfunc(int functype,struct ast *l) {
	struct fncall *n = (struct fncall*)malloc(sizeof(struct fncall));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = 'F';
	n->l = l;
	n->functype = functype;
	return (struct ast*)n;
}

struct ast *newcall(struct symbol *s,struct ast *l) {
	struct ufncall *n = (struct ufncall*)malloc(sizeof(struct ufncall));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = 'C';
	n->l = l;
	n->s = s;
	return (struct ast*)n;
}

struct ast *newref(struct symbol *s) {
	struct symref *n = (struct symref*)malloc(sizeof(struct symref));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = 'N';
	n->s = s;
	return (struct ast*)n;
}

struct ast *newasgn(struct symbol *s,struct ast *v) {
	struct symasgn *n = (struct symasgn*)malloc(sizeof(struct symasgn));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = '=';
	n->s = s;
	n->v = v;
	return (struct ast*)n;
}

struct ast *newflow(int nodetype,struct ast *cond,struct ast *tl,struct ast *tr) {
	struct flow *n = (struct flow*)malloc(sizeof(struct flow));
	if (!n) {
		yyerror("out of space");
		exit(0);
	}
	n->nodetype = nodetype;
	n->tl = tl;
	n->el = tr;
	n->cond = cond;
	return (struct ast*)n;
}

/* define a function */
void dodef(struct symbol *sym,struct symlist *syms,struct ast *stmts) {
	if (sym->syms) symlistfree(sym->syms);
	if (sym->func) treefree(sym->func);
	sym->syms = syms;
	sym->func = stmts;
}

struct ast *newnum(double d) {
	struct numval *n = (struct numval*)malloc(sizeof(struct numval));
	if (!n) {
			perror("out of space!");
			return NULL;
	}
	n->nodetype = 'K';
	n->number = d;
	return (struct ast*)n;
}

static double callbuiltin(struct fncall*);
static double calluser(struct ufncall*);

double eval(struct ast *n) {
	struct flow *f = (struct flow*)n;

	double v;
	switch(n->nodetype) {
	case '+':
		v = eval(n->l) + eval(n->r);break;
	case '-':
		v = eval(n->l) - eval(n->r);break;
	case '*':
		v = eval(n->l) * eval(n->r);break;
	case '/':
		v = eval(n->l) / eval(n->r);break;
	case 'M':
		v = -eval(n->l);break;
	/* name reference */
	case 'N':
		v = ((struct symref*)n)->s->value;break;
	case '|':
		v = eval(n->l); if (v < 0) v = -v;break;
	case '=':
		v = ((struct symasgn*)n)->s->value = eval(((struct symasgn*)n)->v);
		break;
	case 'K':
		v = ((struct numval*)n)->number;break;
	case '1':
		v = (eval(n->l) > eval(n->r)) ? 1.0 : 0.0;break;
	case '2':
		v = (eval(n->l) < eval(n->r)) ? 1.0 : 0.0;break;
	case '3':
		v = (eval(n->l) != eval(n->r)) ? 1.0 : 0.0;break;
	case '4':
		v = (eval(n->l) == eval(n->r)) ? 1.0 : 0.0;break;
	case '5':
		v = (eval(n->l) >= eval(n->r)) ? 1.0 : 0.0;break;
	case '6':
		v = (eval(n->l) <= eval(n->r)) ? 1.0 : 0.0;break;
	case 'I':
		if (eval(f->cond) != 0) {
			if (f->tl) v = eval(f->tl);
			else v = 0.0;
		} else {
			if (f->el) v = eval(f->el);
			else v = 0.0;
		}
		break;
	case 'W':
		v = 0.0;
		if (f->tl) {
			while (eval(f->tl) != 0) {
				v = eval(f->tl);
			}
		}
		break;
	/* list of statement */
	case 'L':
		eval(n->l); v = eval(n->r); break;
	case 'F':
		v = callbuiltin((struct fncall*)n);break;
	case 'C':
		v = calluser((struct ufncall*)n);break;
	default:
		fprintf(stderr,"internal error: bad node %c\n",n->nodetype);
		break;
	}
	return v;
}

void treefree(struct ast *n) {
	switch(n->nodetype) {
	case '+':
	case '-':
	case '*':
	case '/':
	case '1':case '2':case '3':case '4':case '5':case '6':
	case 'L':
		treefree(n->l);
		treefree(n->r);
		break;
	case '|':
	case 'M':
	case 'C':
	case 'F':
		treefree(n->l);
		break;
	case 'K':
	case 'N':
		break;
	case '=':
		free(((struct symasgn*)n)->v);
		break;
	case 'I':
	case 'W':
		free(((struct flow*)n)->cond);
		if (((struct flow*)n)->tl) free(((struct flow*)n)->tl);
		if (((struct flow*)n)->el) free(((struct flow*)n)->el);
		break;
	default:
		fprintf(stderr,"internal error: bad node %c\n",n->nodetype);
		break;
	}
	free(n);
}


struct symlist *newsymlist(struct symbol *sym,struct symlist *next) {
	struct symlist *sl = (struct symlist*)malloc(sizeof(struct symlist));
	if (!sl) {
		perror("out of space!");
		return NULL;
	}
	sl->sym = sym;
	sl->next = next;
	return sl;
}

void symlistfree(struct symlist *sl) {
	struct symlist *p = sl->next,*t;
	while (p) {
		t = p->next;
		free(p);
		p = t;
	}
	free(sl);
}

static unsigned symhash(const char *sym) {
	unsigned hash = 0;
	while (*sym) {
		hash = hash * 9 ^ *sym;
		++sym;
	}
	return hash;
}

struct symbol *lookup(const char *sym) {
	struct symbol *sp = &symtab[symhash(sym) % NHASH];
	int scount = NHASH;

	if (sp->name && strcmp(sp->name,sym) == 0) {
		return sp;
	}

	while (--scount >= 0) {
		if (sp->name && strcmp(sp->name,sym) == 0) return sp;
		if (!sp->name) {
			sp->name = strdup(sym);
			sp->func = NULL;
			sp->syms = NULL;
			return sp;
		}
		sp++;
		if (sp >= symtab + NHASH) {
			sp = symtab;
		}
	}
	fputs("symbol table overflow\n",stderr);
	abort();
}

static double callbuiltin(struct fncall *fn) {
	enum bifs functype = fn->functype;
	double v = eval(fn->l);

	switch (functype) {
	case B_sqrt:
		return sqrt(v);
	case B_exp:
		return exp(v);
	case B_log:
		return log(v);
	case B_print:
		printf("= %4.4g\n",v);
		break;
	default:
		yyerror("Unknow built-in function %d",functype);
		break;
	}
}

static double calluser(struct ufncall *f) {
	struct symbol *fn = f->s; /* function name */
	struct symlist *sl;			 /* dummy arguments */
	struct ast *args = f->l; /* actual arguments */
	double *oldval,*newval;
	double v;
	int nargs;
	int i;

	if (!fn->func) {
		yyerror("call to undefined function",fn->name);
		return 0.0;
	}
	sl = fn->syms;
	for (nargs = 0;sl;sl = sl->next) {
		++nargs;
	}

	oldval = (double*)malloc(nargs * sizeof(double));
	newval = (double*)malloc(nargs * sizeof(double));
	if (!oldval || !newval) {
		yyerror("Out of space in %s\n",fn->name);
		return 0.0;
	}
	for (i = 0; i < nargs; i++) {
		if (!args) {
			yyerror("too few args in call to %s\n",fn->name);
			free(oldval);free(newval);
			return 0.0;
		}

		if (args->nodetype == 'L') {
			newval[i] = eval(args->l);
			args = args->r;
		} else {
			newval[i] = eval(args);
			args = NULL;
		}
	}

	sl = fn->syms;
	for (i = 0; i < nargs; i++) {
		struct symbol *s = sl->sym;
		oldval[i] = s->value;
		s->value = newval[i];
		sl = sl->next;
	}

	free(newval);

	v = eval(fn->func);

	sl = fn->syms;
	for (i = 0; i < nargs; i++) {
		struct symbol *s = sl->sym;
		s->value = oldval[i];
		sl = sl->next;
	}
	free(oldval);
	return v;
}
