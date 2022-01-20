#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>

using namespace llvm;
using namespace std;

static LLVMContext Context;
static Module *g_module = new Module("my compiler", Context);
static vector<string> fun_args;

Function *createFunc(IRBuilder<> &builder, const string &name) {
	Type *u32Ty = Type::getInt32Ty(Context);
	Type *intTy = builder.getInt32Ty();
	Type *ptrTy = intTy->getPointerTo();
	FunctionType *func_type = FunctionType::get(builder.getInt32Ty(), ptrTy, false);
	Function *foo_func = Function::Create(func_type, GlobalValue::ExternalLinkage, name, g_module);
	return foo_func;
}

void setFunArgs(Function *foo_func, const vector<string> &fun_args) {
	unsigned index = 0;
	for (auto ai = foo_func->arg_begin(), end = foo_func->arg_end();
			ai != end; ++ai, ++index) {
		ai->setName(fun_args[index]);
	}
}

BasicBlock *createBB(Function *foo_func, const string &name) {
	return BasicBlock::Create(Context, name, foo_func);
}

Value *getGEP(IRBuilder<> &builder, Value *base, Value *offset) {
	return builder.CreateGEP(builder.getInt32Ty(), base, offset, "a1");
}

int main(int argc, char *argv[]) {
	fun_args.emplace_back("a");
	IRBuilder<> builder{Context};
	Function *foo_func = createFunc(builder, "foo");
	setFunArgs(foo_func, fun_args);

	Value *base = foo_func->arg_begin();
	BasicBlock *entry = createBB(foo_func, "entry");
	builder.SetInsertPoint(entry);
	Value *gep = getGEP(builder, base, builder.getInt32(1));
	builder.CreateRet(builder.getInt32(0));
	assert(!verifyFunction(*foo_func));
	g_module->print(llvm::errs(), nullptr);
	return 0;
}
