#!/usr/bin/python
# -*- coding:utf-8 -*-

from pymtl import *
from sys import argv
from RegIncr import RegIncr

input_values = [int(x, 0) for x in argv[1:]]
input_values.extend([0] * 3)

model = RegIncr()
model.vcd_file = "regincr-sim.vcd"
model.elaborate()

sim = SimulationTool(model)
sim.reset()


for input_value in input_values:
    # Write input value to input port
    model.in_.value = input_value

    # Display input and output ports
    sim.print_line_trace()

    # Tick simulator one cycle
    sim.cycle()
