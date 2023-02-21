RELEASE_TYPE=Release

if [ ! -v SCRIPT_DIR ]; then
	SCRIPT_DIR=$(pwd)
	ROCM_ROOT=$SCRIPT_DIR/..
	ROCM_OUTPUT=$ROCM_ROOT/../output/rocm
fi
