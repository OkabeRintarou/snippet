#!/bin/bash
./sort -nocheck_is_sorted -block_size 4096 -total_memory 10485760 -sub_table_prefix 10M_4K_ input
./sort -nocheck_is_sorted -block_size 4096 -total_memory 52428800 -sub_table_prefix 50M_4K_ input
./sort -nocheck_is_sorted -block_size 4096 -total_memory 104857600  -sub_table_prefix 100M_4K_ input
./sort -nocheck_is_sorted -block_size 16384 -total_memory 52428800 -sub_table_prefix 50M_16K_ input
./sort -nocheck_is_sorted -block_size 65536 -total_memory 52428800 -sub_table_prefix 50M_64K_ input
./sort -nocheck_is_sorted -block_size 131072 -total_memory 52428800 -sub_table_prefix 50M_128K_ input
./sort -nocheck_is_sorted -block_size 262144 -total_memory 52428800 -sub_table_prefix 50M_256K_ input
./sort -nocheck_is_sorted -block_size 524288 -total_memory 52428800 -sub_table_prefix 50M_512K_ input
./sort -nocheck_is_sorted -block_size 10485760 -total_memory 52428800 -sub_table_prefix 50M_1M_ input
./sort -nocheck_is_sorted -block_size 16777216 -total_memory 52428800 -sub_table_prefix 50M_16M_ input

