#!/usr/bin/python3

from twisted.internet import reactor


class Countdown(object):
    counter = 5

    @classmethod
    def count(cls):
        if cls.counter == 0:
            reactor.stop()
        else:
            print(cls.counter, '...')
            cls.counter -= 1
            reactor.callLater(1, cls.count)


reactor.callWhenRunning(Countdown.count)
print('Start')
reactor.run()
print('Stop')
