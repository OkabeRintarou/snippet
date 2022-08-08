#!/usr/bin/python3

from bcc import BPF

BPF(text='int kprobe____x64_sys_clone(void *ctx) { bpf_trace_printk("Hello, World!\\n"); return 0; }').trace_print()
