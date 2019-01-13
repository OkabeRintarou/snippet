LLVM_CONFIG?=llvm-config

ifndef VERBOSE
QUIET:=@
endif

CXX:=clang++
SRC_DIR?=$(PWD)
LDFLAGS+=$(shell $(LLVM_CONFIG) --ldflags)
COMMON_FLAGS=-std=c++17
CXXFALSG+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) -cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)
CLANGLIBS=\
	-Wl,--start-group\
	-lclang\
	-lclangFrontend\
	-lclangDriver\
	-lclangSerialization\
	-lclangParse\
	-lclangSema\
	-lclangAnalysis\
	-lclangEdit\
	-lclangAST\
	-lclangLex\
	-lclangBasic\
	-Wl,--end-group
LLVMLIBS=$(shell $(LLVM_CONFIG) --libs)


PROJECT=myproject
PROJECT_OBJECTS=project.o

TEST_LEXER=test_lexer
TEST_LEXER_OBJECTS=test_lexer.o

TEST_PARSER=test_parser
TEST_PARSER_OBJECTS=test_parser.o

TEST_ALL=test_all
TEST_ALL_OBJECTS=test_all.o

BINS=$(PROJECT) $(TEST_LEXER) $(TEST_PARSER) $(TEST_ALL)
OBJS=$(PROJECT_OBJECTS) $(TEST_LEXER_OBJECTS) $(TEST_PARSER_OBJECTS) $(TEST_ALL_OBJECTS)

default:$(BINS)

%.o:$(SRC_DIR)/%.cc
	@echo Compiling $*.cc
	$(QUIET)$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $<

$(PROJECT):$(PROJECT_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(CLANGLIBS) $(LLVMLIBS)

$(TEST_LEXER):$(TEST_LEXER_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(CLANGLIBS) $(LLVMLIBS)

$(TEST_PARSER):$(TEST_PARSER_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(CLANGLIBS) $(LLVMLIBS)

$(TEST_ALL):$(TEST_ALL_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(CLANGLIBS) $(LLVMLIBS)
.PHONY:clean
clean:
	$(QUIET)rm -f $(BINS) $(OBJS)
