#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace boost::asio;

class printer
{
public:
	printer(io_service& io)
		:timer(io,boost::posix_time::seconds(1)),
		count(0)
	{
		timer.async_wait(boost::bind(&printer::print,this));
	}

	~printer()
	{
		cout << "Final count is " << count;
	}

	void print()
	{
		if(count < 5){
			cout << count << endl;
			++count;
			timer.expires_at(timer.expires_at() + boost::posix_time::seconds(1));
			timer.async_wait(boost::bind(&printer::print,this));
		}
	}
private:
	boost::asio::deadline_timer timer;
	int count;
};

int main()
{
	io_service io;
	printer p(io);
	io.run();
	return 0;
}
