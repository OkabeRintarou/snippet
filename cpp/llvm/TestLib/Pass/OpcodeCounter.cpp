#include <llvm/ADT/StringMap.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace {

using ResultOpcodeCounter = llvm::StringMap<unsigned>;

struct OpcodeCounter : llvm::AnalysisInfoMixin<OpcodeCounter> {
  using Result = ResultOpcodeCounter;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
    return generateOpcodeMap(F);
  }

  static bool isRequired() { return true; }

private:
  static Result generateOpcodeMap(llvm::Function &F);

private:
  static llvm::AnalysisKey Key;
  friend struct llvm::AnalysisInfoMixin<OpcodeCounter>;
};

llvm::AnalysisKey OpcodeCounter::Key;

OpcodeCounter::Result OpcodeCounter::generateOpcodeMap(llvm::Function &F) {
  OpcodeCounter::Result OpcodeMap;

  for (auto &BB : F) {
    for (auto &Inst : BB) {

      StringRef Name = Inst.getOpcodeName();
      if (OpcodeMap.find(Name) == OpcodeMap.end()) {
        OpcodeMap[Name] = 1;
      } else {
        ++OpcodeMap[Name];
      }
    }
  }

  return OpcodeMap;
}

static void printOpcodeCounterResult(llvm::raw_ostream &OS,
                                     const ResultOpcodeCounter &OpcodeMap) {
  OS << "=================================================" << '\n';
  OS << "LLVM-TUTOR: OpcodeCounter results\n";
  OS << "=================================================" << '\n';
  const char *str1 = "OPCODE";
  const char *str2 = "#TIME USED";
  OS << format("%-20s %-10s\n", str1, str2);
  OS << "-------------------------------------------------" << '\n';
  for (auto &I : OpcodeMap) {
    OS << format("%-20s %-10lu\n", I.first().str().c_str(), I.second);
  }
  OS << "-------------------------------------------------" << '\n' << '\n';
}

class OpcodeCounterPrinter : public llvm::PassInfoMixin<OpcodeCounterPrinter> {
  llvm::raw_ostream &OS;

public:
  explicit OpcodeCounterPrinter(llvm::raw_ostream &os) : OS(os) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    const auto &OpcodeMap = FAM.getResult<OpcodeCounter>(F);

    OS << "Printing analysis 'OpcodeCounter Pass' for function '" << F.getName()
       << "'\n";

    printOpcodeCounterResult(OS, OpcodeMap);
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // namespace

PassPluginLibraryInfo getOpcodeCounterPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "OpcodeCounter", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [&](StringRef Name, FunctionPassManager &FPM,
                ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "print<opcode-counter>") {
                FPM.addPass(OpcodeCounterPrinter(llvm::errs()));
                return true;
              }
              return false;
            });
        PB.registerVectorizerStartEPCallback(
            [](llvm::FunctionPassManager &PM, llvm::OptimizationLevel Level) {
              PM.addPass(OpcodeCounterPrinter(llvm::errs()));
            });
        PB.registerAnalysisRegistrationCallback(
            [](FunctionAnalysisManager &FAM) {
              FAM.registerPass([&] { return OpcodeCounter(); });
            });
      }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getOpcodeCounterPluginInfo();
}
