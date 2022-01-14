#include <cassert>
#include <string>
#include <vector>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>

using namespace llvm;
using namespace std;

using BBList = SmallVector<BasicBlock *, 16>;
using ValList = SmallVector<Value *, 16>;

static LLVMContext Context;
static Module *g_module = new Module("my compiler", Context);
static vector<string> fun_args;

Function *createFunc(IRBuilder<> &builder, const string &name) {
	std::vector<Type *> args(fun_args.size(), Type::getInt32Ty(Context));
	FunctionType *func_type = FunctionType::get(builder.getInt32Ty(), args, false);
	Function *foo_func = Function::Create(func_type, GlobalValue::ExternalLinkage, name, g_module);
	return foo_func;
}

void setFuncArgs(Function *foo_func, const vector<string> &fun_args) {
	unsigned idx = 0;
	for (auto it = foo_func->arg_begin(), end = foo_func->arg_end();
			it != end; ++it, ++idx) {
		it->setName(fun_args[idx]);
	}
}

BasicBlock *createBB(Function *foo_func, const string &name) {
	return BasicBlock::Create(Context, name, foo_func);
}

GlobalVariable* createGlob(IRBuilder<> &builder, const std::string &name) {
	g_module->getOrInsertGlobal(name, builder.getInt32Ty());
	GlobalVariable *g_var = g_module->getNamedGlobal(name);
	g_var->setLinkage(GlobalValue::CommonLinkage);
	g_var->setAlignment(MaybeAlign(4));
	return g_var;
}

Value *createArith(IRBuilder<> &builder, Value *lhs, Value *rhs) {
	return builder.CreateMul(lhs, rhs, "multmp");
}

Value *createIfElse(IRBuilder<> &builder, const BBList &bb_list, const ValList &val_list) {
	Value *condtn = val_list[0];
	Value *arg1 = val_list[1];

	BasicBlock* then_bb = bb_list[0];
	BasicBlock *else_bb = bb_list[1];
	BasicBlock *merge_bb = bb_list[2];

	builder.CreateCondBr(condtn, then_bb, else_bb);

	builder.SetInsertPoint(then_bb);
	// %thenaddtmp = a + 1
	Value *then_val = builder.CreateAdd(arg1, builder.getInt32(1), "thenaddtmp");
	builder.CreateBr(merge_bb);

	// elseaddtmp = a + 2
	builder.SetInsertPoint(else_bb);
	Value *else_val = builder.CreateAdd(arg1, builder.getInt32(2), "elseaddtmp");
	builder.CreateBr(merge_bb);

	unsigned phi_bb_size = val_list.size() - 1;
	builder.SetInsertPoint(merge_bb);
	PHINode *phi = builder.CreatePHI(Type::getInt32Ty(Context), phi_bb_size, "iftmp");
	phi->addIncoming(then_val, then_bb);
	phi->addIncoming(else_val, else_bb);
	return phi;
}

Value *createLoop(IRBuilder<> &builder, const BBList &bb_list, const ValList &val_list, 
					Value *start_val, Value *end_val) {
	BasicBlock *preheader_bb = builder.GetInsertBlock();
	Value *val = val_list[0];
	BasicBlock *loop_bb = bb_list[0];
	builder.CreateBr(loop_bb);
	builder.SetInsertPoint(loop_bb);
	// for (i = 1; i < b; i++)
	PHINode *ind_var = builder.CreatePHI(Type::getInt32Ty(Context), 2, "i");
	ind_var->addIncoming(start_val, preheader_bb);
	// a += 5;
	Value *add = builder.CreateAdd(val, builder.getInt32(5), "addtmp");
	Value *step_val = builder.getInt32(1);
	Value *next_val = builder.CreateAdd(ind_var, step_val, "nextval");
	Value *end_cond = builder.CreateICmpULT(ind_var, end_val, "endcond");

	BasicBlock *loop_end_bb = builder.GetInsertBlock();
	BasicBlock *after_bb = bb_list[1];
	builder.CreateCondBr(end_cond, loop_bb, after_bb);
	builder.SetInsertPoint(after_bb);
	ind_var->addIncoming(next_val, loop_end_bb);
	return add;
}

int main(int argc, char *argv[]) {
	IRBuilder<> builder{Context};
	fun_args.push_back("a");
	fun_args.push_back("b");

	GlobalVariable *g_var = createGlob(builder, "x");
	Function *foo_func = createFunc(builder, "foo");
	setFuncArgs(foo_func, fun_args);
	BasicBlock *entry = createBB(foo_func, "entry");
	builder.SetInsertPoint(entry);
	Value *arg1 = foo_func->arg_begin();
	Value *constant = builder.getInt32(16);
	// val1 = a * 16
	Value *val1 = createArith(builder, arg1, constant);
	Value *val2 = builder.getInt32(100);
	// val1 < val2
	Value *compare = builder.CreateICmpSLT(val1, val2, "cmptmp");
	
	// if
	ValList val_list;
	val_list.push_back(compare);
	val_list.push_back(arg1);

	BasicBlock *then_bb = createBB(foo_func, "then");
	BasicBlock *else_bb = createBB(foo_func, "else");
	BasicBlock *merge_bb = createBB(foo_func, "ifcont");
	BBList bb_list;
	bb_list.push_back(then_bb);
	bb_list.push_back(else_bb);
	bb_list.push_back(merge_bb);
	Value *v = createIfElse(builder, bb_list, val_list);

	// for loop
	// for (i = 1; i < b; i++) { a += 5; }
	Value *arg_a = arg1;
	Value *arg_b = arg1 + 1;
	ValList loop_val_list;
	loop_val_list.push_back(arg_a);

	BBList loop_bb_list;
	BasicBlock *loop_bb = createBB(foo_func, "loop");
	BasicBlock *after_bb = createBB(foo_func, "afterloop");
	loop_bb_list.push_back(loop_bb);
	loop_bb_list.push_back(after_bb);

	Value *start_val = builder.getInt32(1);
	Value *res = createLoop(builder, loop_bb_list, loop_val_list, start_val, arg_b);

	builder.CreateRet(v);
	assert(!verifyFunction(*foo_func));
	g_module->print(llvm::errs(), nullptr);
	return 0;
}
