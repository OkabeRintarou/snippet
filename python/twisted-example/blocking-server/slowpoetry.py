#!/usr/bin/python3
import optparse
import os
import socket
import time


def parse_args():
    usage = """usage: %prog [options] poetry-file

This is the Slow Poetry Server, blocking edition.
Run it like this:

  python slowpoetry.py <path-to-poetry-file>

If you are in the base directory of the twisted-intro package,
you could run it like this:

  python blocking-server/slowpoetry.py poetry/ecstasy.txt

to serve up John Donne's Ecstasy, which I know you want to do.
"""
    parser = optparse.OptionParser(usage)

    help = "The port to listen on. Default to random available port."
    parser.add_option('--port', type='int', help=help)

    help = "The interface to listen on. Default is localhost."
    parser.add_option('--iface', help=help, default='localhost')

    help = "The number of seconds between sending bytes."
    parser.add_option('--delay', type='float', help=help, default=7)

    help = "The number of bytes to send at a time."
    parser.add_option('--num-bytes', type='int', help=help, default=10)

    option, args = parser.parse_args()

    if len(args) != 1:
        parser.error('Provide exactly one poetry file')

    poetry_file = args[0]

    if not os.path.exists(poetry_file):
        parser.error('No such file: %s' % poetry_file)

    return option, poetry_file


def send_poetry(sock, poetry_file, num_bytes, delay):
    """Send some poetry slowly down the socket"""
    inputf = open(poetry_file, 'r')

    while True:
        data = inputf.read(num_bytes)

        if not data:
            sock.close()
            inputf.close()
            return

        print('Sending %d bytes' % (len(data),))

        try:
            sock.sendall(bytes(data,encoding='utf-8'))  # this is a blocking call
        except socket.error:

            sock.close()
            inputf.close()

            return
        time.sleep(delay)


def serve(listen_socket, poetry_file, num_bytes, delay):
    while True:
        sock, addr = listen_socket.accept()

        print('Somebody at %s wants poetry!' % (addr,))

        send_poetry(sock, poetry_file, num_bytes, delay)

        print('Close connection between %s.' % (addr,))


def main():
    option, poetry_file = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((option.iface, option.port or 0))

    sock.listen(5)

    print('Serving %s on port %s.' % (poetry_file, sock.getsockname()[1]))

    serve(sock, poetry_file, option.num_bytes, option.delay)


if __name__ == '__main__':
    main()
