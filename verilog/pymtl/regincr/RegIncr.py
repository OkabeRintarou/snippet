#!/usr/bin/python
# -*- coding:utf-8 -*-

from pymtl import *


class RegIncr(Model):
    def __init__(self):
        # Port-based interface
        self.in_ = InPort(Bits(8))
        self.out = OutPort(Bits(8))

        # Concurrent block modeling register
        self.reg_out = Wire(Bits(8))

        @self.tick
        def block1():

            if self.reset:
                self.reg_out.next = 0
            else:
                self.reg_out.next = self.in_

        @self.combinational
        def block2():
            self.out.value = self.reg_out + 1

    def line_trace(self):
        return 'in:{} ({}) out:{}'.format(self.in_, self.reg_out, self.out)
