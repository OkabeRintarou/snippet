#include <iostream>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace boost::asio;

void print(const boost::system::error_code&,
		asio::deadline_timer* t,int* count)
{
	if(*count < 5){
		cout << "Hello,World" << endl;
		++(*count);
		t->expires_at(t->expires_at() + boost::posix_time::seconds(1));
		t->async_wait(boost::bind(print,boost::asio::placeholders::error,t,count));
	}
}

int main()
{
	io_service io;
	deadline_timer t(io,posix_time::seconds(5));
	int count = 0;
	t.async_wait(boost::bind(print,boost::asio::placeholders::error,&t,&count));
	
	io.run();
	cout << "Final count is " << count << endl;
	return 0;
}
