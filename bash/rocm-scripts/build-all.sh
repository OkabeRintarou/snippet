TARGETS=(libhsakmt llvm rocr rocm-device-libs comgr rocclr)


CUR_DIR=$(pwd)

for t in ${TARGETS[@]};do
	cd $CUR_DIR && bash "build-${t}.sh"
done
