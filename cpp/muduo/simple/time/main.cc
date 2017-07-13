#include <muduo/net/EventLoop.h>
#include <muduo/base/Logging.h>
#include "time.h"

using namespace muduo;
using namespace muduo::net;

int main()
{
	LOG_INFO << "pid = " << getpid();	

	EventLoop loop;
	TimeServer server(&loop,InetAddress(2009));
	server.start();
	loop.loop();
}
