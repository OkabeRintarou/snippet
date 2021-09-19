#define BOOST_ASIO_NO_DEPRECATED
#include <sys/resource.h>
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <boost/asio.hpp>

namespace asio = boost::asio;
using namespace boost::asio::ip;
using boost::system::error_code;

tcp::endpoint addr(make_address("0.0.0.0"), 43333);
int thread_count = 4;
int qdata = 4;
std::atomic_uint64_t session_count{0};

void usage(const char *pg_name) {
	printf("\n	Usage: %s [ThreadCount] [QueryDataLength]\n", pg_name);
	printf("\n	Default: %s %d 4\n", pg_name, thread_count);
	printf("\n	For example:\n			%s 2 32\n", pg_name);
	printf("\n	That's means: start server with 2 threads, and per data-package is 32 bytes.\n\n");
}

class tcp_connection : public std::enable_shared_from_this<tcp_connection> {
public:
	using pointer = std::shared_ptr<tcp_connection>;

	static pointer create(asio::io_context &io) {
		return pointer(new tcp_connection(io));
	}

	tcp::socket &socket() {
		return socket_;
	}
	void start();

private:
	explicit tcp_connection(asio::io_context &io)
		:socket_(io) {}

	void handle_writed(const error_code e, size_t n);
	void handle_read(const error_code e, size_t n);
	void handle_error();
private:
	tcp::socket socket_;
	char buf_[4];
};

void tcp_connection::start() {
	socket_.async_read_some(asio::buffer(buf_, qdata),
			std::bind(&tcp_connection::handle_read, shared_from_this(), std::placeholders::_1, std::placeholders::_2));
}

void tcp_connection::handle_error() {

}

void tcp_connection::handle_read(const error_code e, size_t n) {
	if (e) {
		handle_error();
		return;
	}

	asio::async_write(socket_, asio::buffer(buf_, n),
			std::bind(&tcp_connection::handle_writed, shared_from_this(),
				std::placeholders::_1, std::placeholders::_2));
}

void tcp_connection::handle_writed(const error_code e, size_t n) {
	if (e) {
		handle_error();
		return;
	}
	start();
}

class tcp_server {
public:
	explicit tcp_server(asio::io_context &io) :
		io_(io),
		acceptor_(io, addr) {
		start_accept();
	}

private:
	void start_accept();

	void handle_accept(tcp_connection::pointer new_connection, const error_code e);

private:
	asio::io_context &io_;
	tcp::acceptor acceptor_;
};

void tcp_server::start_accept() {
	auto new_connection = tcp_connection::create(io_);

	acceptor_.async_accept(new_connection->socket(),
			std::bind(&tcp_server::handle_accept, this, new_connection, std::placeholders::_1));
}

void tcp_server::handle_accept(tcp_connection::pointer new_connection, const error_code e) {
	if (!e) {
		new_connection->start();
	}
	start_accept();
}

int main(int argc, char *argv[]) {
	if (argc > 1) {
		if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
			usage(argv[0]);
			exit(1);
		}
	}

	if (argc > 1) {
		thread_count = atoi(argv[1]);
	}
	if (argc > 2) {
		qdata = atoi(argv[2]);
	}

	rlimit of = {65536, 65536};
	if (-1 == setrlimit(RLIMIT_NOFILE, &of)) {
		perror("setrlimit");
		exit(1);
	}

	printf("startup server, thread:%d, qdata:%d, listen %s:%d\n", thread_count,
			qdata, addr.address().to_string().c_str(), addr.port());

	asio::io_context io_context;
	tcp_server server(io_context);

	std::vector<std::thread> threads;
	for (int n = 0; n < thread_count; n++) {
		threads.emplace_back([&] {
			io_context.run();
		});
	}

	for (auto &thread : threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}
	return 0;
}
