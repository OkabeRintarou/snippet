#!/usr/bin/python3
import traceback
from twisted.internet import reactor


def stack():
    print('The python stack')
    traceback.print_stack()


reactor.callWhenRunning(stack)
print('Starting the reactor.')
reactor.run()
