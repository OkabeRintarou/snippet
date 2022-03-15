BIN=sum
ISPC=ispc
CXX=g++
CXXFLAGS=-O2
ISPCFLAGS=-O2 --target=sse4-x2 --arch=x86-64
sum:sum.o sum_ispc.o
	$(CXX) $(CXXFLAGS) -o sum sum.o sum_ispc.o

sum.o:sum.cpp sum_ispc.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

sum_ispc.h sum_ispc.o:sum.ispc
	$(ISPC) $(ISPCFLAGS) $< -o sum_ispc.o -h sum_ispc.h

.PHONY:clean
clean:
	@rm *.o sum_ispc.h sum
