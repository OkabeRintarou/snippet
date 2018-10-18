#!/usr/bin/python
# -*- coding:utf-8 -*-

from pymtl import *
from RegIncr import RegIncr


class RegIncr2stage(Model):
    def __init__(self):
        self.in_ = InPort(Bits(8))
        self.out = OutPort(Bits(8))

        self.reg_incr_0 = RegIncr()

        self.connect(self.in_, self.reg_incr_0.in_)

        self.reg_incr_1 = RegIncr()

        self.connect(self.reg_incr_0.out, self.reg_incr_1.in_)
        self.connect(self.reg_incr_1.out, self.out)

    def line_trace(self):
        return "{} ({}) {}".format(
            self.in_,
            self.reg_incr_0.line_trace(),
            self.reg_incr_1.line_trace(),
            self.out
        )
