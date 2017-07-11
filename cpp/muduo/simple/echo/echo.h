#ifndef _ECHO_SERVER_H
#define _ECHO_SERVER_H

#include <muduo/net/TcpServer.h>

class EchoServer
{
public:
	EchoServer(muduo::net::EventLoop* loop,
				const muduo::net::InetAddress& listenAddr);
	void start(); // class server_.start()
private:
	void onConnection(const muduo::net::TcpConnectionPtr& conn);
	void onMessage(const muduo::net::TcpConnectionPtr& conn,
					muduo::net::Buffer* buf,
					muduo::Timestamp time);
	muduo::net::EventLoop* loop_;
	muduo::net::TcpServer server_;
};

#endif
