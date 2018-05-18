#!/usr/bin/python3
import datetime
import errno
import optparse
import select
import socket


def parse_args():
    usage = """usage: %prog [options] [hostname]:port ...

This is the Get Poetry Now! client, asynchronous edition.
Run it like this:

  python get-poetry.py port1 port2 port3 ...

If you are in the base directory of the twisted-intro package,
you could run it like this:

  python async-client/get-poetry.py 10001 10002 10003

to grab poetry from servers on ports 10001, 10002, and 10003.

Of course, there need to be servers listening on those ports
for that to work.
"""

    parser = optparse.OptionParser(usage)

    _, addresses = parser.parse_args()

    if not addresses:
        print(parser.format_help())
        parser.exit()

    def parse_address(addr):
        if ':' not in addr:
            host = '127.0.0.1'
            port = addr
        else:
            host, port = addr.split(':', 1)

        if not port.isdigit():
            parser.error('Port must be integers')
        return host, int(port)

    return list(map(parse_address, addresses))


def connect(address):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(address)
    sock.setblocking(0)
    return sock


def format_address(addr):
    host, port = addr
    return '%s:%s' % (host or '127.0.0.1', port)


def get_poetry(sockets):
    """Download poetry from all the given sockets"""

    poems = dict.fromkeys(sockets, '')  # socket -> accumulated poem

    # socket -> task number
    sock2task = dict([(s, i + 1) for i, s in enumerate(sockets)])

    sockets = list(sockets)  # make a copy

    # we go around this loop until we'va gotten all the poetry
    # from all the sockets. This is a `reactor loop`
    while sockets:
        rlist, _, _ = select.select(sockets, [], [])

        # rlist is the list of sockets with data ready to read
        for sock in rlist:
            data = ''

            while True:
                try:
                    new_data = sock.recv(1024)
                except socket.error as e:
                    if e.args[0] == errno.EWOULDBLOCK:
                        break
                    raise
                else:
                    if not new_data:
                        break
                    data += new_data.decode('utf-8')

            task_num = sock2task[sock]

            if not data:
                sockets.remove(sock)
                sock.close()
                print('Task %d finished' % (task_num,))
            else:
                addr_fmt = format_address(sock.getpeername())
                msg = 'Task %d: got %d bytes of poetry from %s'
                print(msg % (task_num, len(data), addr_fmt))

            poems[sock] += data

    return poems


def main():
    addresses = parse_args()

    start = datetime.datetime.now()

    sockets = list(map(connect, addresses))

    poems = get_poetry(sockets)

    elapsed = datetime.datetime.now() - start

    for i, sock in enumerate(sockets):
        print('Task: %d: %d bytes of poetry' % (i + 1, len(poems[sock])))

    print('Got %d poems in %s' % (len(addresses), elapsed))


if __name__ == '__main__':
    main()
