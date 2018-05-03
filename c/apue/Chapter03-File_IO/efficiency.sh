#!/bin/bash

BUFS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288)
for b in ${BUFS[@]};do
	gcc -D${b} -o efficiency efficiency.c
done
