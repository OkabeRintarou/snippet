#!/usr/bin/python3

from socketserver import ThreadingTCPServer
from socketserver import BaseRequestHandler

class EchoHandler(BaseRequestHandler):
    def handle(self):
        print('got connection from ', self.client_address)
        while True:
            data = self.request.recv(4096)
            if data:
                send  = self.request.send(data)
            else:
                print('disconnect',self.client_address)
                self.request.close()
                break


if __name__ == '__main__':
    listen_addr = ('0.0.0.0',2017)
    server = ThreadingTCPServer(listen_addr,EchoHandler)
    server.serve_forever()
    
