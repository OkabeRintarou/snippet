#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <boost/asio.hpp>

namespace asio = boost::asio;
using namespace asio::ip;
using namespace std::chrono;
using boost::system::error_code;

tcp::endpoint addr(make_address("0.0.0.0"), 43333);
const int g_buf_cnt = 4096;
size_t thread_count = 4;
size_t conn_count = 1000;
size_t qdata = 32;
std::atomic_uint64_t session_count;
std::atomic_uint64_t g_sendbytes{0}, g_recvbytes{0}, g_qps{0};

static void usage(const char *const pg_name) {
	printf("\n	Usage: %s [ThreadCount] [ConnectionCount] [QueryDataLength]\n", pg_name);
	printf("\n	Default: %s %lu 1024 4\n", pg_name, thread_count);
	printf("\n	For example:\n		%s 2 1000 32\n", pg_name);
	printf("\n	That's means ctart chient with 2 threads, "
			"create 1000 tcp connection to server, and per data-package is 32 bytes.\n\n");
}

static void show_status() {
	static int show_title = 0;
	static long long last_qps = 0;
	static unsigned long long last_sendbytes = 0, last_recvbytes = 0;
	static auto start_time = system_clock::now();
	static auto last_time = system_clock::now();

	auto now = system_clock::now();
	using std::cout;
	using std::endl;
	using std::setw;
	const size_t w = 10;
	const size_t ww = 20;
	const float MB = static_cast<float>(1024 * 1024);

	if (show_title++ % 10 == 0) {
		cout << "thread: " << thread_count << ", qdata:" << qdata << endl;
		cout << setw(w) << "conn" << setw(w) << "send(MB)" << setw(w) << "recv(MB)"
			<< setw(w) << "qps" << setw(ww) << "AverageQps" << setw(ww) << "time_delta(ms)" << endl;
	}
	float send_mb = static_cast<float>(g_sendbytes - last_sendbytes) / MB;
	float recv_mb = static_cast<float>(g_recvbytes - last_recvbytes) / MB;
	uint64_t qps = g_qps - last_qps;
	float average_qps = static_cast<float>(g_recvbytes) / static_cast<float>(qdata) /
		static_cast<float>(1.0f, duration_cast<seconds>(now - start_time).count() + 1);
	size_t time_delta = static_cast<size_t>(duration_cast<milliseconds>(now - last_time).count());

	cout << setw(w) << session_count << setw(w) << send_mb << setw(w) << recv_mb
		<< setw(w) << qps << setw(ww) << average_qps << setw(ww) << time_delta << endl;

	last_time = now;
	last_sendbytes = g_sendbytes;
	last_recvbytes = g_recvbytes;
	last_qps = g_qps;
}

class tcp_client {
public:
	explicit tcp_client(asio::io_context &io) : 
		io_(io),
   		socket_(io)	{
		start_client();
	}
private:
	void start_client();
	void handle_connected(const error_code e);
	void handle_writed(const error_code e, size_t n);
	void handle_read(const error_code e, size_t n);
	void handle_error();


	void async_write();
private:
	asio::io_context &io_;
	tcp::socket socket_;
	char write_buf_[g_buf_cnt];
};

void tcp_client::start_client() {
	socket_.async_connect(addr, std::bind(&tcp_client::handle_connected, this, std::placeholders::_1));
}

void tcp_client::async_write() {
	asio::async_write(socket_, asio::buffer(write_buf_, qdata), 
			std::bind(&tcp_client::handle_writed, this, std::placeholders::_1, std::placeholders::_2));
}

void tcp_client::handle_connected(const error_code e) {
	if (e) {
		return;
	}
	++session_count;
	async_write();
}

void tcp_client::handle_writed(const error_code e, std::size_t n) {
	if (e) {
		handle_error();
		return;
	}
	g_qps++;
	g_sendbytes += n;
	socket_.async_read_some(asio::buffer(write_buf_, sizeof(write_buf_)),
			std::bind(&tcp_client::handle_read, this, std::placeholders::_1, std::placeholders::_2));
}

void tcp_client::handle_read(const error_code e, std::size_t n) {
	if (e) {
		handle_error();
		return;
	}
	g_qps++;
	g_recvbytes += n;
	async_write();
}

void tcp_client::handle_error() {
	--session_count;
}

int main(int argc, char *argv[]) {
	if (argc > 1) {
		if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
			usage(argv[0]);
			exit(1);
		}
	}
	std::ios::sync_with_stdio(false);

	if (argc > 1) {
		thread_count = atoi(argv[1]);
	}
	if (argc > 2) {
		conn_count = atoi(argv[2]);
	}
	if (argc > 3) {
		qdata = atoi(argv[3]);
	}
	rlimit of = {65536, 65536};
	if (-1 == setrlimit(RLIMIT_NOFILE, &of)) {
		perror("setrlimit");
		exit(1);
	}

	asio::io_context io;

	std::vector<std::unique_ptr<tcp_client>> clients;
	for (int i = 0; i < conn_count; i++) {
		clients.emplace_back(std::make_unique<tcp_client>(io));
	}

	std::vector<std::thread> threads;
	for (int i = 0; i < thread_count; i++) {
		threads.emplace_back([&io] {
			io.run();
		});
	}

	std::atomic_bool quit{false};

	std::thread status_thread([&quit] {
		for (;;) {
			if (quit.load()) {
				std::cerr << "Asio client quit..." << std::endl;
				break;
			}
			sleep(1);
			show_status();
		}
	});

	for (auto &t : threads) {
		t.join();
	}
	quit.store(true);
	status_thread.join();
	return 0;
}
