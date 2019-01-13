#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include <iostream>
#include <string>
#include <memory>

using namespace llvm;

static cl::opt<std::string> FileName(cl::Positional, cl::desc("Bitcode file"), cl::Required);

int main(int argc, char *argv[]) {

	cl::ParseCommandLineOptions(argc, argv, "LLVM hello world\n");
	LLVMContext context;
	auto mb = MemoryBuffer::getFile(FileName);

	raw_os_ostream cerr(std::cerr);
	raw_os_ostream cout(std::cout);
	if (std::error_code ec = mb.getError()) {
		cerr << "Error reading bitcode: " << ec.message() << '\n';
		return -1;
	}
	auto m = parseBitcodeFile(*mb.get(), context);
	if (!m) {
		cerr << "Error reading bitcode: " << m.takeError() << '\n';
		return -1;
	}

	for (auto &it : (*m)->getFunctionList()) {
		if (!it.isDeclaration()) {
			cout << it.getName() << " has " << it.size() << " basic block(s).\n";
		}
	}
	return 0;
}
