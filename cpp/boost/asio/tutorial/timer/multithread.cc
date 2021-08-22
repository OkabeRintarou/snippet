#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace boost::asio;

class printer {
public:
    printer(io_service &io)
        : strand_(make_strand(io)),
          timer1_(io, asio::chrono::seconds(1)),
          timer2_(io, asio::chrono::seconds(1)),
          count_(0) {

        timer1_.async_wait(bind_executor(strand_, boost::bind(&printer::print1, this)));
        timer2_.async_wait(bind_executor(strand_, boost::bind(&printer::print2, this)));
    }

    ~printer() {
        cout << "Final count is " << count_ << endl;
    }

    void print1() {
        if (count_ < 10) {
            cout << "Timer 1: " << count_ << endl;
            ++count_;
            timer1_.expires_at(timer1_.expiry() + asio::chrono::seconds(1));
            timer1_.async_wait(bind_executor(strand_, boost::bind(&printer::print1, this)));
        }
    }

    void print2() {
        if (count_ < 10) {
            cout << "Timer 2: " << count_ << endl;
            ++count_;
            timer2_.expires_at(timer2_.expiry() + asio::chrono::seconds(1));
            timer2_.async_wait(bind_executor(strand_, boost::bind(&printer::print2, this)));
        }
    }

private:
    strand <io_context::executor_type> strand_;
    steady_timer timer1_;
    steady_timer timer2_;
    int count_;
};

int main() {
    io_context io;
    printer p(io);
    boost::thread t(boost::bind(&io_context::run, &io));
    io.run();
    t.join();
    return 0;
}
