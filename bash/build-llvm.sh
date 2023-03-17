#!/bin/bash
# example: ./build-llvm.sh -i ~/opensource/llvm-project/llvm -o ~/output/llvm/release/ -t /usr/local

usage() {
	echo "Usage: ${0} [-i|--input-dir] [-o|--output-dir] [-t|--install-dir]" 1>&2
	exit 1
}

while [[ $# -gt 0 ]];do
	key=${1}
	case ${key} in
		-i|--input-dir)
			INPUT_DIR=${2}
			shift 2
			;;
		-o|--output-dir)
			OUTPUT_DIR=${2}
			shift 2
			;;
		-t|--install-dir)
			INSTALL_DIR=${2}
			shift 2
			;;
		*)
			usage
			shift
			;;
	esac
done

if [ -z "${INPUT_DIR}" ];then
	usage
	exit 1
fi

if [ -z "${OUTPUT_DIR}" ];then
	usage
	exit 1
fi

if [ -z "${INSTALL_DIR}" ];then
	usage
	exit 1
fi

if [ ! -d "${INPUT_DIR}" ];then
	echo "Input dir ${INPUT_DIR} doesn't exist!!!"
	exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ];then
	echo "Ouput dir ${OUTPUT_DIR} doesn't exist!!!"
	exit 1
fi
if [ ! -d "${INSTALL_DIR}" ];then
	echo "Install dir ${INSTALLD_RI} doesn't exist!!!"
	exit 1
fi


set -e

cd ${OUTPUT_DIR} && cmake -G "Ninja" -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_USE_LINKER=lld -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD="X86;AVR;AMDGPU" -DBUILD_SHARED_LIBS=ON -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_USE_NEWPM=ON -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" "${INPUT_DIR}"

cd "${OUTPUT_DIR}" && ninja all
