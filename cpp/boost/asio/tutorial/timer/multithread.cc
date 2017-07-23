#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace boost::asio;

class printer
{
public:
	printer(io_service& io)
		:strand(io),
		timer1(io,posix_time::seconds(1)),
		timer2(io,posix_time::seconds(1)),
		count(0)
	{
		timer1.async_wait(strand.wrap(boost::bind(&printer::print1,this)));
		timer2.async_wait(strand.wrap(boost::bind(&printer::print2,this)));
	}

	~printer()
	{
		cout << "Final count is " << count << endl;
	}

	void print1()
	{
		if(count < 10){
			cout << "print1 count = " << count << endl;
			++count;
			timer1.expires_at(timer1.expires_at() + posix_time::seconds(1));
			timer1.async_wait(strand.wrap(boost::bind(&printer::print1,this)));
		}
	}

	void print2()
	{
		if(count < 10) {
			cout << "print2 count = " << count << endl;
			++count;
			timer2.expires_at(timer2.expires_at() + posix_time::seconds(1));
			timer2.async_wait(strand.wrap(boost::bind(&printer::print2,this)));
		}
	}
private:
	boost::asio::strand strand;
	boost::asio::deadline_timer timer1;
	boost::asio::deadline_timer timer2;
	int count;
	
};
int main()
{
	io_service io;
	printer p(io);
	io.run();
	return 0;
}
