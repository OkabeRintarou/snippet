#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/experimental/as_single.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/write.hpp>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

namespace asio = boost::asio;
using asio::awaitable;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable_t;
using asio::experimental::as_single_t;
using asio::ip::tcp;
using default_token = as_single_t<use_awaitable_t<>>;
using tcp_socket = default_token::as_default_on_t<tcp::socket>;
namespace this_coro = asio::this_coro;

#define MESSAGE "Hello, World!"

awaitable<void> client(tcp_socket socket) {
    char read[64];

    if (co_await socket.async_connect(
            tcp::endpoint(asio::ip::make_address("127.0.0.1"), 55555)),
        socket.is_open()) {
        auto [e1, nwritten] =
            co_await socket.async_send(asio::buffer(MESSAGE, sizeof(MESSAGE)));
        if (e1) {
            std::cerr << "send message error: " << e1.message() << '\n';
            co_return;
        }
        auto [e2, nread] = co_await socket.async_read_some(asio::buffer(read));
        if (e2) {
            std::cerr << "receive message error: " << e2.message() << '\n';
            co_return;
        }
        assert(nwritten == sizeof(MESSAGE) && nwritten == nread);
    }
}

int main() {
    asio::io_context ctx;
    tcp::socket socket(ctx);
    co_spawn(ctx, client(std::move(socket)), detached);
    ctx.run();
    return 0;
}
