all: product_test.byte library_test.byte
product_test.byte:product_test.ml
	ocamlbuild -pkg oUnit $@
library_test.byte:library_test.ml
	ocamlbuild -pkg oUnit $@
	
.PHONY:clean
clean:
	ocamlbuild -clean
