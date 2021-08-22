#include <string>
#include <memory>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>

using namespace boost;
using namespace boost::asio;

using asio::ip::tcp;

std::string make_daytime_string() {
    using namespace std;
    time_t now = time(0);
    return ctime(&now);
}

class tcp_connection : public std::enable_shared_from_this<tcp_connection> {
public:
    using pointer = std::shared_ptr<tcp_connection>;

    static pointer create(io_context &io) {
        return pointer(new tcp_connection(io));
    }

    tcp::socket &socket() {
        return socket_;
    }

    void start() {
        msg_ = make_daytime_string();

        asio::async_write(socket_, asio::buffer(msg_),
                          boost::bind(&tcp_connection::handle_write, shared_from_this(),
                                      asio::placeholders::error, asio::placeholders::bytes_transferred));
    }

private:
    tcp_connection(io_context &io) : socket_(io) {}

    void handle_write(const boost::system::error_code &, size_t) {
    }

private:
    tcp::socket socket_;
    std::string msg_;
};

class tcp_server {
public:
    explicit tcp_server(io_context &io) :
        io_(io),
        acceptor_(io, tcp::endpoint(tcp::v4(), 13)) {
        start_accept();
    }

private:
    void start_accept();

    void handle_accept(tcp_connection::pointer new_connection, const boost::system::error_code &e);
private:
    io_context &io_;
    tcp::acceptor acceptor_;
};

void tcp_server::start_accept() {
    auto new_connection = tcp_connection::create(io_);

    acceptor_.async_accept(new_connection->socket(),
                           boost::bind(&tcp_server::handle_accept, this, new_connection, asio::placeholders::error));
}

void tcp_server::handle_accept(tcp_connection::pointer new_connection, const boost::system::error_code &e) {
    if (!e) {
        new_connection->start();
    }
    start_accept();
}

int main() {
    try {
        io_context io;
        tcp_server server(io);
        io.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
