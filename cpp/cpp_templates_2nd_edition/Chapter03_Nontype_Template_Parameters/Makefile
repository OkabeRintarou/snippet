targets=stacknontype addvalue stackauto
all:$(targets)

stacknontype:stacknontype.cpp
	g++ -std=c++11 -o $@ $<

addvalue:addvalue.cpp	
	g++ -std=c++11 -o $@ $<

stackauto:stackauto.cpp	
	clang++ -std=c++17 -o $@ $<

.PHONY:clean
clean:
	-rm $(targets)
