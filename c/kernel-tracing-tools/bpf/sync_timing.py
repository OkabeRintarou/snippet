#!/usr/bin/python3

from bcc import BPF

prog="""
#include <uapi/linux/ptrace.h>

struct data_t {
    u64 ts;
    u64 delta_ms;
};

BPF_HASH(last);
BPF_PERF_OUTPUT(events);

#define NS_PER_SEC (1000000000)

int do_trace(struct pt_regs *ctx) {
    struct data_t data = {};
    u64 ts, *tsp, delta, key = 0;

    ts = bpf_ktime_get_ns();
    tsp = last.lookup(&key);
    if (tsp != NULL) {
        delta = ts - *tsp;
        if (delta < NS_PER_SEC) {
            data.ts = ts;
            data.delta_ms = delta / 1000000;
            events.perf_submit(ctx, &data, sizeof(data));
        }
        last.delete(&key);
    }

    // update stored timestamp
    last.update(&key, &ts);
    return 0;
}
"""

b = BPF(text=prog)
b.attach_kprobe(event=b.get_syscall_fnname("sync"), fn_name="do_trace")
print("Tracing for quick sync's... Ctrl-C to end")

start = 0
def print_event(cpu, data, size):
    global start
    event = b["events"].event(data)
    ts = event.ts

    if start == 0:
        start = event.ts
    ts = ts - start
    print("At time %.2f s: multiple syncs detected, last %d ms ago" % (ts / 1000000000.0, event.delta_ms))

b["events"].open_perf_buffer(print_event)
while True:
    b.perf_buffer_poll()
