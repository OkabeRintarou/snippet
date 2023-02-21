. common.sh
ROCM_LLVM_OUTPUT=$ROCM_OUTPUT/llvm

mkdir -p $ROCM_LLVM_OUTPUT && cd $ROCM_LLVM_OUTPUT

cmake -DCMAKE_INSTALL_PREFIX=$ROCM_LLVM_OUTPUT -DCMAKE_BUILD_TYPE=$RELEASE_TYPE -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" $ROCM_ROOT/llvm-project/llvm
make -j12
make install

