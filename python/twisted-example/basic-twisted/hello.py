#!/usr/bin/python3
from twisted.internet import reactor


def hello():
    print('Hello from the reactor loop')
    print('Lately I feel like I\'m stuck in a rut.')


reactor.callWhenRunning(hello)
print('Starting the reactor.')
reactor.run()
