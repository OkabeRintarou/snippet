#!/usr/bin/python
# -*- coding:utf-8 -*-

from pymtl import *
from RegIncr import RegIncr


def test_basic(dump_vcd):
    model = RegIncr()
    model.vcd_file = dump_vcd
    model.elaborate()

    sim = SimulationTool(model)
    sim.reset()

    def t(in_, out):
        model.in_.value = in_
        sim.eval_combinational()
        sim.print_line_trace()
        if out != '?':
            assert model.out == out
        sim.cycle()

    t(0x00, '?')
    t(0x13, 0x01)
    t(0x27, 0x14)
    t(0x00, 0x28)
    t(0x00, 0x01)
    t(0x00, 0x01)
