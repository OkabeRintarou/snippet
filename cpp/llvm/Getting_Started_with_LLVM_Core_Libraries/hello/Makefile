LLVM_CONFIG?=llvm-config

ifndef VERBOSE
QUIET:=@
endif

CXX:=clang++
SRC_DIR?=$(PWD)
LDFLAGS+=$(shell $(LLVM_CONFIG) --ldflags)
COMMON_FLAGS=-Wall -Wextra -std=c++17
CXXFALSG+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) -cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

HELLO=helloworld
HELLO_OBJECTS=hello.o


default:$(HELLO)

#hello.o:hello.c
#	clang++ -c $(CPPFLAGS) $(CXXFLAGS) $<
%.o:$(SRC_DIR)/%.cc
	@echo Compiling $*.cc
	$(QUIET)$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $<

$(HELLO):$(HELLO_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ `$(LLVM_CONFIG) --libs bitreader core support`

.PHONY:clean
clean:
	$(QUIET)rm -f $(HELLO) $(HELLO_OBJECTS)
