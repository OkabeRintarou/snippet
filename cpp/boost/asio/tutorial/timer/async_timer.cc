#include <iostream>

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace boost::asio;

void print(const boost::system::error_code&)
{
	cout << "Hello,World" << endl;
}

int main()
{
	io_service io;
	deadline_timer t(io,posix_time::seconds(5));
	t.async_wait(print);
	io.run();
	return 0;
}
