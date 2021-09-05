#include <cstddef>
#include <optional>
#include <memory>
#include <functional>
#include <string>
#include <queue>
#include <unordered_set>
#include <boost/asio.hpp>

namespace asio = boost::asio;
using tcp = asio::ip::tcp;
using error_code = boost::system::error_code;
using namespace std::placeholders;

using message_handler = std::function<void (std::string)>;
using error_handler = std::function<void ()>;

class session : public std::enable_shared_from_this<session> {
public:
    using pointer = std::shared_ptr<session>;
public:
    explicit session(tcp::socket &&sock);

    void start(message_handler&& on_message, error_handler&& on_error);
    void post(const std::string& message);
private:
    void async_read();
    void async_write();

    void on_read(error_code error, std::size_t bytes_transferred);
    void on_write(error_code error, std::size_t bytes_transferred);
private:
    tcp::socket socket_;
    asio::streambuf stream_buf_;
    std::queue<std::string> outgoing_;
    message_handler on_message_;
    error_handler on_error_;
};

session::session(tcp::socket &&sock) : socket_(std::move(sock)) {}

void session::start(message_handler &&on_message, error_handler &&on_error) {
    this->on_message_ = std::move(on_message);
    this->on_error_ = std::move(on_error);
    async_read();
}

void session::post(const std::string &message) {
    bool idle = outgoing_.empty();
    outgoing_.emplace(message);

    if (idle) {
        async_write();
    }
}

void session::async_read() {
    asio::async_read_until(socket_, stream_buf_, "\n",
                           std::bind(&session::on_read, shared_from_this(), _1, _2));
}

void session::async_write() {
    asio::async_write(socket_, asio::buffer(outgoing_.front()),
                      std::bind(&session::on_write, shared_from_this(), _1, _2));
}

void session::on_read(error_code error, std::size_t bytes_transferred) {
    if (!error) {
        std::stringstream message;
        message << socket_.remote_endpoint(error) << ": " << std::istream(&stream_buf_).rdbuf();
        stream_buf_.consume(bytes_transferred);
        on_message_(message.str());
        async_read();
    } else {
        socket_.close(error);
        on_error_();
    }
}

void session::on_write(error_code error, std::size_t bytes_transferred) {
    if (!error) {
        outgoing_.pop();
        if (!outgoing_.empty()) {
            async_write();
        }
    } else {
        socket_.close(error);
        on_error_();
    }
}

class server {
public:
    server(asio::io_context& io, std::uint16_t port);

    void async_accept();
private:
    void post(const std::string& message);
private:
    asio::io_context &context_;
    tcp::acceptor acceptor_;
    std::optional<tcp::socket> socket_;
    std::unordered_set<session::pointer> clients_;
};

server::server(asio::io_context &io, std::uint16_t port)
    :context_(io),
     acceptor_(io, tcp::endpoint(tcp::v4(), port)) {

}

void server::async_accept() {
    socket_.emplace(context_);

    acceptor_.async_accept(*socket_, [&](const error_code& error) {
        auto client = std::make_shared<session>(std::move(*socket_));

        client->post("Welcome to chat\n\r");
        post("We have a newcomer\n\r");
        clients_.insert(client);

        client->start(std::bind(&server::post, this, _1),
                      [&, weak = std::weak_ptr(client)] {
            if (auto shared = weak.lock(); shared && clients_.erase(shared)) {
                post("We are one less\n\r");
            }
        });
        async_accept();
    });
}

void server::post(const std::string &message) {
    for (auto& client : clients_) {
        client->post(message);
    }
}

int main() {
    asio::io_context io_context;
    server srv(io_context, 15001);
    srv.async_accept();
    io_context.run();
    return 0;
}
