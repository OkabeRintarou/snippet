BINS=hsa_copy hsa_add vec_add.gfx1030.co vec_add.gfx1030.o

INC_PATH ?= /opt/rocm/include
LIB_PATH ?= /opt/rocm/lib
CLANG ?= /opt/rocm/llvm/bin/clang

all: $(BINS)

hsa_copy:hsa_copy.cc
	g++ -std=c++17 -I$(INC_PATH) -L$(LIB_PATH) -o $@ $< -lhsa-runtime64

hsa_add:hsa_add.cc
	g++ -g -std=c++17 -I$(INC_PATH) -L$(LIB_PATH) -o $@ $< -lhsa-runtime64

vec_add.gfx1030.o:vec_add.gfx1030.s
	$(CLANG) -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1030 -c -o $@ $<

vec_add.gfx1030.co:vec_add.gfx1030.o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=gfx1030 -o $@ $<
	

clean:
	rm $(BINS)
