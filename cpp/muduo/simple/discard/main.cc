#include <muduo/net/EventLoop.h>
#include <muduo/base/Logging.h>
#include "discard.h"

using namespace muduo;
using namespace muduo::net;

int main()
{
	LOG_INFO << "pid = " << getpid();	

	EventLoop loop;
	DiscardServer server(&loop,InetAddress(2009));
	server.start();
	loop.loop();
}
