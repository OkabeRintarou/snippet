#include <ctime>
#include "time.h"

#include <muduo/base/Logging.h>

#include <boost/bind.hpp>

using namespace muduo;
using namespace muduo::net;

TimeServer::TimeServer(EventLoop* loop,const InetAddress& listenAddr)
	:server_(loop,listenAddr,"DiscardServer")
{
	server_.setConnectionCallback(
		boost::bind(&TimeServer::onConnection,this,_1));
	server_.setMessageCallback(
		boost::bind(&TimeServer::onMessage,this,_1,_2,_3));
}

void TimeServer::start()
{
	server_.start();
}

void TimeServer::onConnection(const TcpConnectionPtr& conn)
{
	LOG_INFO << "DiscardServer - " << conn->peerAddress().toIpPort() << " -> "
			<< conn->localAddress().toIpPort() << " is "
			<< (conn->connected() ? "UP" : "DOWN");
	if(conn->connected()){
		time_t now = ::time(NULL);
		int32_t be32 = sockets::hostToNetwork32(static_cast<int32_t>(now));
		conn->send(&be32,sizeof(be32));
		conn->shutdown();
	}
}

void TimeServer::onMessage(const TcpConnectionPtr& conn,Buffer *buf,Timestamp time)
{
	string msg(buf->retrieveAllAsString());
	LOG_INFO << conn->name() << " discards " << msg.size()
			<< " bytes received at " << time.toString();
}


