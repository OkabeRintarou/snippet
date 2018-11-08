#!/bin/bash
#####################################
# See more: https://apt.llvm.org
#####################################

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main"
sudo apt-get update

# install just clang,lld and lldb(6.0 release)
sudo apt-get install -y clang-6.0 lldb-6.0 lld-6.0

# install all key packages
# LLVM
# apt-get install libllvm-6.0-ocaml-dev libllvm6.0 llvm-6.0 llvm-6.0-dev llvm-6.0-doc llvm-6.0-examples llvm-6.0-runtime
# Clang and co
# apt-get install clang-6.0 clang-tools-6.0 clang-6.0-doc libclang-common-6.0-dev libclang-6.0-dev libclang1-6.0 clang-format-6.0 python-clang-6.0
# libfuzzer
# apt-get install libfuzzer-6.0-dev
# lldb
# apt-get install lldb-6.0
# lld (linker)
# apt-get install lld-6.0



