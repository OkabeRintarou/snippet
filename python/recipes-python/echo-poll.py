#!/usr/bin/python3

import socket
import select


if __name__ == '__main__':
    listen_address = ('0.0.0.0',2017)
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server_socket.bind(listen_address)
    server_socket.listen(5)
    # server_socket.setblocking(0)
    poll = select.poll()
    poll.register(server_socket.fileno(),select.POLLIN)

    connections = {}
    while True:
        events = poll.poll(10000) # ten seconds
        for fileno,event in events:
            if fileno == server_socket.fileno():
                (client_socket,client_address) = server_socket.accept()
                print('got connection from',client_address)
                # client_socket.setblocking(0)
                poll.register(client_socket.fileno(),select.POLLIN)
                connections[client_socket.fileno()] = client_socket
            elif event & select.POLLIN:
                client_socket = connections[fileno]
                data = client_socket.recv(4096)
                if data:
                    client_socket.send(data)
                else:
                    poll.unregister(fileno)
                    client_socket.close()
                    del connections[fileno]
        
