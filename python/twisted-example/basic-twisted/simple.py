#!/usr/bin/python3

from twisted.internet import pollreactor

pollreactor.install()

from twisted.internet import reactor

if __name__ == '__main__':
    reactor.run()
