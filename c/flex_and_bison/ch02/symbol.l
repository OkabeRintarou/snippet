%option noyywrap nodefault yylineno case-insensitive

%{

	struct symbol {
		char *name;
		struct ref *reflist;
	};

	struct ref {
		struct ref *next;
		char *filename;
		int flags;
		int lineno;
	};

	#define NHASH 9997
	struct symbol symtab[NHASH];

	struct symbol *lookup(const char*);
	void addref(int,const char *,const char *,int);
	static void printrefs();

	char *curfilename;
%}
%%

a |
an |
and |
are |
as |
at |
be |
but |
for |
in |
is |
it |
of |
on |
or |
that |
the |
this |
to


[a-z]+(\'(s|t))? { addref(yylineno,curfilename,yytext,0); }
.|\n  ;

%%

int main(int argc,char *argv[]) {
	int i;

	if (argc < 2) {
		curfilename = "(stdin)";
		yylineno = 1;
		yylex();
	} else {
		for (i = 1; i < argc; i++) {
			FILE *f = fopen(argv[i],"r");
			if (!f) {
				perror(argv[1]);
				return 1;
			}
			curfilename = argv[i];

			yyrestart(f);
			yylineno = 1;
			yylex();
			fclose(f);
		}
	}
	printrefs();
	return 0;
}

/* hash a symbol */
static unsigned symhash(const char *sym) {
	unsigned int hash = 0;
	unsigned c;
	while(c = *sym++) hash = hash * 9 ^ c;
	return hash;
}


struct symbol *lookup(const char *sym) {
	struct symbol *sp = &symtab[symhash(sym) % NHASH];
	int scount = NHASH;

	while (--scount >= 0) {
		if (sp->name && !strcmp(sp->name,sym)) return sp;
		if (!sp->name) {
			sp->name = strdup(sym);
			sp->reflist = 0;
			return sp;
		}
		if (++sp >= symtab) sp = symtab;
	}
	fputs("symbol table overflow\n",stderr);
	abort();
}

void addref(int lineno,const char *filename,const char *word,int flags) {
	struct ref *r;
	struct symbol *sp = lookup(word);

	if (sp->reflist && sp->reflist->lineno == lineno &&
			strcmp(sp->reflist->filename,filename) == 0) {
		return;
	}

	r = (struct ref*)malloc(sizeof(struct ref));
	if (!r) {
		fputs("out of space!\n",stderr);
		abort();
	}
	r->next = sp->reflist;
	r->filename = filename;
	r->lineno = lineno;
	r->flags = flags;
	sp->reflist = r;
}

static int symcompare(const void *xa, const void *xb) {
	const struct symbol *a = (struct symbol*)xa;
	const struct symbol *b = (struct symbol*)xb;

	if (!a->name) {
		if (!b->name) return 0;
		return 1;
	}
	if (!b->name) return -1;
	return strcmp(a->name,b->name);
}

void printrefs() {
	struct symbol *sp;
	qsort(symtab,NHASH,sizeof(struct symbol),symcompare);
	for (sp = symtab;sp->name && sp < symtab + NHASH; sp++) {
		char *prevfn = NULL;

		/* reverse the list of references */
		struct ref *rp = sp->reflist;
		struct ref *rpp = NULL; /* previous ref */
		struct ref *rpn;

		do {
			rpn = rp->next;
			rp->next = rpp;
			rpp = rp;
			rp = rpn;
		} while (rp);

		printf("%10s",sp->name);
		for (rp = rpp; rp ;rp = rp->next) {
			if (rp->filename == prevfn) {
				printf(" %d",rp->lineno);
			} else {
				printf(" %s:%d",rp->filename,rp->lineno);
				prevfn = rp->filename;
			}
		}
		printf("\n");
	}
}
